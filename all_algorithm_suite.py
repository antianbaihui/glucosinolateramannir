"""Bounded, reproducible nested-CV comparison of requested algorithm families."""
from optimize_single_modalities import *
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pybaselines import Baseline
OUT=HERE/'all_algorithm_suite';OUT.mkdir(exist_ok=True)

def inputs(cohort):
    from compare_algorithms import data
    nx,nw,ny,nt,nids=data();nx=nx[:88];nids=nids[:88]
    mask=(nw>=4000)&(nw<=7500);nx=nx[:,mask];nw=nw[mask]
    if cohort=='nir':return [nx],[nw],ny,nids
    cache=np.load(HERE/'fusion_comparison'/'paired_preprocessed.npz');lab=pd.read_csv(HERE/'fusion_comparison'/'paired_sample_labels.csv',dtype={'ID':str});ids=lab.ID.to_numpy();y=lab.label.to_numpy()
    rw=cache['raman_uniform_wavenumbers'];raw=np.array([np.interp(rw,cache['raman_original_wavenumbers'],r) for r in cache['raman_raw']]);base=Baseline(x_data=rw)
    corrected=[]
    for row in raw:
        b,info=base.airpls(row,lam=1e6,max_iter=100,tol=.001,normalize_weights=True);assert info['tol_history'][-1]<=.001;corrected.append(row-b)
    rx=np.array(corrected)
    if cohort=='raman':return [rx],[rw],y,ids
    ix=np.array([np.flatnonzero(nids==sid)[0] for sid in ids]);assert np.array_equal(ny[ix],y)
    if cohort=='nir_paired':return [nx[ix]],[nw],y,ids
    return [nx[ix],rx],[nw,rw],y,ids

def prep_blocks(raw,w,fit,val,recipe,cohort):
    blocks=[]
    for j,(x,ww) in enumerate(zip(raw,w)):
        isnir=cohort.startswith('nir') or (cohort=='fusion' and j==0)
        if isnir:
            a,b=nir_preprocess(x[fit],x[val],'snv' if recipe==0 else 'msc',1 if recipe==0 else 2,float(np.median(np.diff(ww))))
        else:
            def transform(z):
                sm=savgol_filter(z,9,2,axis=1);den=np.trapezoid(abs(sm),ww,axis=1)
                return (sm if recipe==0 else savgol_filter(z,9,2,deriv=1,delta=ww[1]-ww[0],axis=1))/den[:,None]
            a,b=transform(x[fit]),transform(x[val])
        sc=StandardScaler().fit(a);a=sc.transform(a);b=sc.transform(b)
        blocks.append((a,b))
    return blocks

def block_concat(blocks):
    return tuple(np.concatenate([z[j]/np.sqrt(np.var(z[0],axis=0).sum()) for z in blocks],axis=1) for j in [0,1])

def opls(a,b,y,north):
    # One predictive component; remove sequential y-orthogonal score components.
    a=a.copy();b=b.copy();yc=y-y.mean()
    for _ in range(north):
        w=a.T@yc;norm=np.linalg.norm(w)
        if norm<1e-12:break
        w=w/norm;t=a@w;p=a.T@t/(t@t);wo=p-w*(w@p);norm=np.linalg.norm(wo)
        if norm<1e-12:break
        wo=wo/norm;to=a@wo;den=to@to
        if den<1e-12:break
        assert abs(to@yc)<=1e-7*max(1,np.linalg.norm(to)*np.linalg.norm(yc))
        po=a.T@to/den;a-=np.outer(to,po);b-=np.outer(b@wo,po)
    return PLSRegression(n_components=1,scale=False,max_iter=1000).fit(a,y).predict(b).ravel()

def spls(a,b,y,k,keep):
    # Soft-thresholded NIPALS PLS1, X and y deflation; binary response => sPLS-DA.
    a=a.copy();b=b.copy();res=y-y.mean();pred=np.full(len(b),float(y.mean()));supports=[]
    for _ in range(k):
        c=a.T@res;keepn=min(keep,len(c));cut=0 if keepn==len(c) else np.partition(abs(c),-keepn-1)[-keepn-1]
        weight=np.sign(c)*np.maximum(abs(c)-cut,0);norm=np.linalg.norm(weight)
        if norm<1e-12:break
        weight/=norm;supports.append(np.flatnonzero(weight));t=a@weight;den=t@t
        if den<1e-12:break
        loading=a.T@t/den;q=res@t/den;tb=b@weight;pred+=tb*q
        a-=np.outer(t,loading);b-=np.outer(tb,loading);res-=t*q
    return pred,supports

def cars(a,y,seed):
    """30 MC/EDF/ARS steps, training-only 3-fold RMSE choice of subset.
    EDF decays from p to 2. ARS samples with replacement, retaining unique indexes.
    PLS2 calibration; Monte Carlo uses 80% of current training observations.
    """
    rng=np.random.default_rng(seed);active=np.arange(a.shape[1]);p=a.shape[1];subsets=[]
    for step in range(30):
        rows=rng.choice(len(y),max(6,int(.8*len(y))),replace=False)
        m=PLSRegression(n_components=min(2,len(active)),scale=False,max_iter=1000).fit(a[rows][:,active],y[rows]);importance=abs(m.coef_.ravel())+1e-15
        keep=max(2,int(round(p*np.exp(-np.log(p/2)*step/29))))
        if keep<len(active):
            top=np.argsort(-importance)[:keep];active=active[top];importance=importance[top]
        if step>0 and len(active)>2:
            indexes=np.unique(rng.choice(len(active),size=len(active),replace=True,p=importance/importance.sum()))
            if len(indexes)<2:indexes=np.argsort(-importance)[:2]
            active=active[indexes]
        subsets.append(active.copy())
    inner=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(a,y));errors=[]
    for cols in subsets:
        pred=np.zeros(len(y))
        for it,iv in inner:pred[iv]=PLSRegression(n_components=min(2,len(cols)),scale=False,max_iter=1000).fit(a[it][:,cols],y[it]).predict(a[iv][:,cols]).ravel()
        errors.append(np.mean((pred-y)**2))
    chosen=subsets[int(np.argmin(errors))];assert len(chosen)>=2 and len(np.unique(chosen))==len(chosen)
    return chosen

def representations(blocks,y,seed):
    a,b=block_concat(blocks);sc=StandardScaler().fit(a);za=sc.transform(a);zb=sc.transform(b)
    # PCA/SVM need equal per-feature scales; multi-block PLS preserves block factors.
    pca=PCA(n_components=min(10,len(y)-1),svd_solver='full').fit(za);pc=(pca.transform(za),pca.transform(zb))
    pls=PLSRegression(n_components=4,scale=False,max_iter=1000).fit(a,y);ps=(pls.transform(a),pls.transform(b))
    ss=StandardScaler().fit(ps[0]);ps=(ss.transform(ps[0]),ss.transform(ps[1]))
    W=pls.x_weights_;T=pls.x_scores_;Q=pls.y_loadings_.ravel();contrib=np.sum(T*T,axis=0)*Q*Q
    vip=np.sqrt(a.shape[1]*((W*W/np.sum(W*W,axis=0))@contrib)/max(contrib.sum(),1e-20));vi=np.flatnonzero(vip>1)
    if len(vi)<2:vi=np.argsort(-vip)[:2]
    selected=[];offset=0
    for j,(aa,bb) in enumerate(blocks):
        selected.extend((cars(aa,y,seed+17*j)+offset).tolist());offset+=aa.shape[1]
    ci=np.array(selected,dtype=int)
    out={'full':(a,b),'PCA':pc,'PLS':ps,'VIP':(a[:,vi],b[:,vi]),'CARS':(a[:,ci],b[:,ci]),'metadata':dict(CARS_indices=ci.tolist(),VIP_indices=vi.tolist())}
    # Standardize selected predictors for SVM; scale still fitted only on training.
    ss=StandardScaler().fit(out['CARS'][0]);out['CARS_SVM']=(ss.transform(out['CARS'][0]),ss.transform(out['CARS'][1]))
    kernels=[]
    for aa,bb in blocks:
        m=PCA(n_components=min(10,len(y)-1),svd_solver='full').fit(aa);aa=m.transform(aa);bb=m.transform(bb);ss=StandardScaler().fit(aa);kernels.append((ss.transform(aa),ss.transform(bb)))
    out['blocks']=kernels;return out

def configurations(cohort):
    cfg={}
    cfg['PLS_DA']=[dict(k=k,threshold=h) for k in [2,4,8] for h in [.35,.5]]
    cfg['OPLS_DA']=[dict(north=n,threshold=h) for n in [0,1,2,4] for h in [.35,.5]]
    cfg['sPLS_DA']=[dict(k=k,keep=v,threshold=h) for k in [2,4] for v in [10,30,100] for h in [.35,.5]]
    for family in ['RandomForest','ExtraTrees']:
        cfg[family]=[dict(rep=r,depth=d,leaf=l) for r in ['full','PCA'] for d,l in [(3,3),(None,2)]]
    for family in ['XGBoost','LightGBM']:
        cfg[family]=[dict(rep=r,depth=d) for r in ['PCA','PLS'] for d in [1,2]]
    for family in ['VIP_XGBoost','CARS_XGBoost']:
        cfg[family]=[dict(rep=family.split('_')[0],depth=d) for d in [1,2]]
    cfg['PCA_KNN']=[dict(k=k,weights=z) for k in [3,5,9] for z in ['uniform','distance']]
    for family in ['SVM_RBF','CARS_SVM']:
        cfg[family]=[dict(C=c,gamma=g) for c in [.1,1,10] for g in [.1,1]]
    if cohort=='fusion':cfg['MultiKernel_SVM']=[dict(C=c,gamma=g,alpha=h) for c in [1,10] for g in [.1,1] for h in [0,.25,.5,.75,1]]
    return cfg

def predict(family,cfg,reps,y,seed):
    a,b=reps['full']
    if family in ['OPLS_DA','sPLS_DA','PLS_DA']:
        if family=='OPLS_DA':s=opls(a,b,y,cfg['north'])
        elif family=='sPLS_DA':s,_=spls(a,b,y,cfg['k'],cfg['keep'])
        else:s=PLSRegression(n_components=cfg['k'],scale=False,max_iter=1000).fit(a,y).predict(b).ravel()
        return (s>=cfg['threshold']).astype(int),s
    if family=='MultiKernel_SVM':
        train=[];test=[]
        for aa,bb in reps['blocks']:
            gamma=cfg['gamma']/(aa.shape[1]*aa.var());train.append(rbf_kernel(aa,aa,gamma=gamma));test.append(rbf_kernel(bb,aa,gamma=gamma))
        h=cfg['alpha'];a=h*train[0]+(1-h)*train[1];b=h*test[0]+(1-h)*test[1]
        model=SVC(kernel='precomputed',C=cfg['C'],class_weight='balanced').fit(a,y);return model.predict(b),model.decision_function(b)
    if family in ['SVM_RBF','CARS_SVM']:
        a,b=reps['PLS' if family=='SVM_RBF' else 'CARS_SVM'];gamma=cfg['gamma']/(a.shape[1]*a.var());model=SVC(C=cfg['C'],gamma=gamma,class_weight='balanced').fit(a,y);return model.predict(b),model.decision_function(b)
    if family=='PCA_KNN':a,b=reps['PCA'];model=KNeighborsClassifier(n_neighbors=cfg['k'],weights=cfg['weights']).fit(a,y)
    else:
        a,b=reps[cfg['rep']]
        if family in ['RandomForest','ExtraTrees']:
            cls=RandomForestClassifier if family=='RandomForest' else ExtraTreesClassifier
            model=cls(n_estimators=100,max_depth=cfg['depth'],min_samples_leaf=cfg['leaf'],max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=1).fit(a,y)
        elif family=='LightGBM':model=LGBMClassifier(n_estimators=80,learning_rate=.03,max_depth=cfg['depth'],num_leaves=2**cfg['depth'],min_child_samples=5,reg_lambda=5,verbosity=-1,class_weight='balanced',random_state=seed,n_jobs=1).fit(a,y)
        else:model=XGBClassifier(n_estimators=80,learning_rate=.03,max_depth=cfg['depth'],min_child_weight=3,reg_lambda=5,subsample=.8,colsample_bytree=.8,scale_pos_weight=float((y==0).sum()/(y==1).sum()),random_state=seed,n_jobs=1,tree_method='hist').fit(a,y)
    s=model.predict_proba(b)[:,1];return (s>=.5).astype(int),s

def checks():
    rng=np.random.default_rng(26);a=rng.normal(size=(40,20));b=rng.normal(size=(5,20));a-=a.mean(axis=0);y=rng.normal(size=40)
    expected=PLSRegression(n_components=3,scale=False).fit(a,y).predict(b).ravel();actual,_=spls(a,b,y,3,20);assert np.allclose(actual,expected,atol=1e-7)
    _,supports=spls(a,b,y,3,5);assert all(len(s)<=5 for s in supports)
    opls(a,b,y,4);selected=cars(a,(y>np.median(y)).astype(int),42);assert len(selected)>=2
    print('OPLS orthogonality, sPLS dense-limit equivalence/sparsity, CARS index checks passed.',flush=True)

def main(cohort):
    folder=OUT/cohort;folder.mkdir(exist_ok=True);raw,w,y,ids=inputs(cohort);cfgs=configurations(cohort);innern=4 if cohort=='nir' else 3
    protocol=dict(cohort=cohort,seeds=SEEDS,outer=5,inner=innern,preprocessing=['NIR4000-7500 SNV+SG1 / Raman airPLS1e6+SG0+area','NIR4000-7500 MSC+SG2 / Raman airPLS1e6+SG1+area'],configurations=cfgs,CARS='30 MC80% steps, EDF p-to-2, coefficient-proportional ARS with replacement; choose minimum training-internal3fold RMSE, PLS2; separate per modality in fusion',sPLS='soft-thresholded NIPALS PLS1 weights, X/y deflation; keep10/30/100 per component; not claimed identical to mixOmics or Chun-Keles refit',OPLS='0/1/2/4 sequential y-orthogonal components, one predictive component',fusion='Training-scaled blocks divided by sqrt(total training variance), feature concatenation; CARS per block; multi-kernel weighted sum of per-block PCA10 RBF kernels, weights0/.25/.5/.75/1',scope='Bounded grids; fixed total-GS extreme labels; no auxiliary middle samples in these direct classifiers; repeated exploration requires future external validation')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8');rows=[];preds=[];splits=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for repeat,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(raw[0],y)):
                inner=list(StratifiedKFold(innern,shuffle=True,random_state=seed+fold+1).split(raw[0][tr],y[tr]));best={name:(-np.inf,None) for name in cfgs};inner_records=[]
                for recipe in [0,1]:
                    scores={name:np.zeros(len(cs)) for name,cs in cfgs.items()}
                    for ii,(it,iv) in enumerate(inner):
                        fit=tr[it];val=tr[iv];assert not set(fit)&set(np.r_[val,te]);reps=representations(prep_blocks(raw,w,fit,val,recipe,cohort),y[fit],seed+100*fold+ii)
                        for name,cs in cfgs.items():
                            for j,cfg in enumerate(cs):
                                pp,_=predict(name,cfg,reps,y[fit],seed+fold);scores[name][j]+=(np.mean(pp[y[val]==0]==0)+np.mean(pp[y[val]==1]==1))/(2*innern)
                    for name,cs in cfgs.items():
                        for j,cfg in enumerate(cs):
                            score=float(scores[name][j]);conf=dict(recipe=recipe,parameters=cfg);inner_records.append(dict(model=name,inner_BA=score,config=json.dumps(conf,sort_keys=True)))
                            if score>best[name][0]+1e-12:best[name]=(score,conf)
                repscache={}
                for name,(score,cfg) in best.items():
                    r=cfg['recipe']
                    if r not in repscache:repscache[r]=representations(prep_blocks(raw,w,tr,te,r,cohort),y[tr],seed+100*fold+99)
                    reps=repscache[r];pp,ss=predict(name,cfg['parameters'],reps,y[tr],seed+fold)
                    auditcfg=dict(cfg)
                    if name.startswith(('CARS','VIP')):auditcfg['selected_features']=reps['metadata'][('CARS' if name.startswith('CARS') else 'VIP')+'_indices']
                    append_result(rows,preds,name,repeat,fold,te,y,ids,pp,ss,score,auditcfg)
                # Family selection is also evaluated without looking at outer results.
                name=max(best,key=lambda v:best[v][0]);winner=rows[-len(cfgs):][list(cfgs).index(name)];ww=[p for p in preds if p['model']==name and p['repeat']==repeat and p['fold']==fold]
                rows.append(dict(winner,model='nested_family_selection',selected_family=name));preds.extend(dict(p,model='nested_family_selection') for p in ww)
                pd.DataFrame(inner_records).to_csv(folder/f'inner_r{repeat}_f{fold}.csv',index=False);pd.DataFrame(rows).to_csv(folder/'fold_metrics.csv',index=False);pd.DataFrame(preds).to_csv(folder/'oof_predictions.csv',index=False)
                splits.append(dict(repeat=repeat,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]));(folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
                print(f'{cohort} {repeat+1}/{fold+1}; elapsed {time.time()-start:.0f}s; nested={winner["BA"]:.3f}',flush=True)
        assert not any(issubclass(v.category,ConvergenceWarning) for v in caught)
        counts={}
        for v in caught:
            key=f'{v.category.__name__}: {v.message}';counts[key]=counts.get(key,0)+1
        (folder/'warnings.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
    print(cohort+' complete',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='checks':checks()
    else:main(sys.argv[1])
