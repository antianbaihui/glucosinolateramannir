"""Further nested-CV NIR/Raman tests with fixed cohorts and fixed outer splits.
python optimize_single_modalities.py nir | raman | checks
"""
import os,sys,json,time,warnings,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'single_modality_optimization';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(HERE/'deps'))
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,matthews_corrcoef,confusion_matrix
from sklearn.exceptions import ConvergenceWarning

SEEDS=[42,2026,87]
BANDS={'full':(3999,10002),'4000_7500':(4000,7500),'4040_5810':(4040,5810)}
SCATTER=['none','snv','msc']
RCONFIG=[dict(k=k,log=log) for k in [2,5,8,12] for log in [False,True]]
SELECT=[dict(method='all',count=0,start_rank=0)]+[dict(method='SPA',count=n,start_rank=r) for n in [10,20,30] for r in [0,1,2]]

def metric(y,p,s):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(BA=balanced_accuracy_score(y,p),AUC=roc_auc_score(y,s),MCC=matthews_corrcoef(y,p),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/max(tp+fp,1),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def append_result(rows,preds,name,rep,fold,te,y,ids,p,s,score,cfg):
    row=dict(model=name,repeat=rep,fold=fold,inner_BA=float(score),config=json.dumps(cfg,sort_keys=True),**metric(y[te],p,s));rows.append(row)
    preds.extend(dict(model=name,repeat=rep,fold=fold,ID=str(ids[i]),y=int(y[i]),prediction=int(pp),score=float(ss)) for i,pp,ss in zip(te,p,s))
    return row

def msc_fit_transform(a,b):
    ref=a.mean(axis=0);rc=ref-ref.mean();den=rc@rc
    assert den>1e-20,'MSC reference is flat'
    def transform(x):
        means=x.mean(axis=1);slopes=((x-means[:,None])@rc)/den
        assert np.all(abs(slopes)>1e-10),'MSC slope is near zero'
        offsets=means-slopes*ref.mean()
        return (x-offsets[:,None])/slopes[:,None]
    return transform(a),transform(b),ref

def nir_preprocess(a,b,scatter,derivative,delta):
    if scatter=='snv':
        a=(a-a.mean(axis=1,keepdims=True))/a.std(axis=1,keepdims=True,ddof=1)
        b=(b-b.mean(axis=1,keepdims=True))/b.std(axis=1,keepdims=True,ddof=1)
    elif scatter=='msc':a,b,_=msc_fit_transform(a,b)
    return savgol_filter(a,15,2,deriv=derivative,delta=delta,axis=1),savgol_filter(b,15,2,deriv=derivative,delta=delta,axis=1)

def spa_paths(a,target,max_count=30):
    """SPA projection core, with three training-correlation-ranked initial variables.

    Limited-start implementation: deliberately not an exhaustive all-start SPA search.
    Length and start rank are selected by the enclosing inner CV, never by test labels.
    """
    z=StandardScaler().fit_transform(a);yc=target-target.mean()
    norms=np.linalg.norm(z,axis=0);corr=abs(z.T@yc)/(np.maximum(norms,1e-20)*np.linalg.norm(yc))
    starts=np.argsort(-corr,kind='stable')[:3];paths=[]
    for start in starts:
        residual=z.copy();chosen=[];index=int(start)
        for step in range(max_count):
            chosen.append(index);v=residual[:,index].copy();den=v@v
            assert den>1e-14,'SPA rank exhausted before requested count'
            residual-=np.outer(v,(v@residual)/den)
            residual[:,chosen]=0
            if step+1<max_count:index=int(np.argmax(np.sum(residual*residual,axis=0)))
        assert len(set(chosen))==max_count
        paths.append(np.array(chosen,dtype=int))
    return paths

def reg_predict(a,b,target,cfg):
    yy=np.log1p(target) if cfg['log'] else target
    m=PLSRegression(n_components=cfg['k'],scale=True,max_iter=1000).fit(a,yy)
    s=m.predict(b).ravel();return np.expm1(s) if cfg['log'] else s

def choose_threshold(score,y,inner,midpoint):
    u=np.unique(score);thresholds=np.r_[u[0]-1,(u[:-1]+u[1:])/2,u[-1]+1];values=np.zeros(len(thresholds))
    for _,iv in inner:
        pp=score[iv,None]>=thresholds[None,:];yy=y[iv]
        values+=(pp[yy==1].mean(axis=0)+(~pp[yy==0]).mean(axis=0))/(2*len(inner))
    winners=np.flatnonzero(values>=values.max()-1e-12)
    i=winners[np.argmin(abs(thresholds[winners]-midpoint))]
    return float(values[i]),float(thresholds[i])

def read_nir():
    # Reuse the previously verified loader and exact row order; function is read-only.
    from compare_algorithms import data
    return data()

def run_nir():
    x,w,y,t,ids=read_nir();mid=np.arange(88,126);delta=float(np.median(np.diff(w)))
    crops={name:x[:,(w>=lo)&(w<=hi)] for name,(lo,hi) in BANDS.items()}
    local=[dict(selector=s,regression=r) for s in SELECT for r in RCONFIG if s['method']=='all' or r['k']<=s['count']]
    protocol={'seeds':SEEDS,'outer_folds':5,'inner_folds':4,'bands':BANDS,'scatter':SCATTER,'SG':{'window':15,'poly':2,'derivatives':[0,1,2]},'selectors':SELECT,'regression':RCONFIG,'candidate_count':len(BANDS)*len(SCATTER)*3*len(local),'SPA':'Successive orthogonal residual projection. Three starts: top three absolute training correlations with RAW total; 10/20/30 predictors. Limited-start variant, not full all-start SPA and not CARS. Selector fit includes only training extremes and auxiliary middle samples.','MSC':'reference mean learned from regression training fold including middle samples, before SG','threshold':'max mean inner BA over inner OOF predicted-content midpoints; ties nearest midpoint of outer-training low/high target means','cohort':'original 88 extremes, 38 middle samples auxiliary training only; no cohort redefinition'}
    (OUT/'nir_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    names=['NIR_none_full','NIR_SNV_full','NIR_MSC_full','NIR_scatter_selected','NIR_SPA_selected','NIR_joint_selected']
    rows=[];preds=[];splits=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for rep,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x[:88],y)):
                inner=list(StratifiedKFold(4,shuffle=True,random_state=seed+fold+1).split(x[tr],y[tr]))
                best={n:(-np.inf,None) for n in names};inner_rows=[]
                midpoint=(t[tr[y[tr]==0]].mean()+t[tr[y[tr]==1]].mean())/2
                for band,raw in crops.items():
                    for scatter in SCATTER:
                        for derivative in [0,1,2]:
                            oof=np.zeros((len(local),len(tr)))
                            for it,iv in inner:
                                fit=np.r_[tr[it],mid];val=tr[iv]
                                assert not set(fit)&set(np.r_[val,te])
                                a,b=nir_preprocess(raw[fit],raw[val],scatter,derivative,delta)
                                paths=spa_paths(a,t[fit]);sets={}
                                for selector in SELECT:
                                    key=(selector['method'],selector['count'],selector['start_rank'])
                                    sets[key]=np.arange(a.shape[1]) if selector['method']=='all' else paths[selector['start_rank']][:selector['count']]
                                for j,cfg in enumerate(local):
                                    s=cfg['selector'];chosen=sets[(s['method'],s['count'],s['start_rank'])]
                                    oof[j,iv]=reg_predict(a[:,chosen],b[:,chosen],t[fit],cfg['regression'])
                            for j,cfg in enumerate(local):
                                score,threshold=choose_threshold(oof[j],y[tr],inner,midpoint)
                                config=dict(band=band,scatter=scatter,derivative=derivative,**cfg,threshold=threshold)
                                inner_rows.append(dict(inner_BA=score,config=json.dumps(config,sort_keys=True)))
                                if cfg['selector']['method']=='all':groups=[{'none':'NIR_none_full','snv':'NIR_SNV_full','msc':'NIR_MSC_full'}[scatter],'NIR_scatter_selected','NIR_joint_selected']
                                else:groups=['NIR_SPA_selected','NIR_joint_selected']
                                for group in groups:
                                    if score>best[group][0]+1e-12:best[group]=(score,config)
                # Evaluate only training-selected candidates on the held-out samples.
                outer_cache={}
                for name,(score,cfg) in best.items():
                    key=json.dumps(cfg,sort_keys=True)
                    if key not in outer_cache:
                        fit=np.r_[tr,mid];a,b=nir_preprocess(crops[cfg['band']][fit],crops[cfg['band']][te],cfg['scatter'],cfg['derivative'],delta)
                        sel=cfg['selector'];chosen=np.arange(a.shape[1]) if sel['method']=='all' else spa_paths(a,t[fit])[sel['start_rank']][:sel['count']]
                        ss=reg_predict(a[:,chosen],b[:,chosen],t[fit],cfg['regression']);pp=(ss>=cfg['threshold']).astype(int)
                        cc=dict(cfg)
                        if sel['method']=='SPA':
                            lo,hi=BANDS[cfg['band']];cc['selected_wavenumbers']=w[(w>=lo)&(w<=hi)][chosen].tolist()
                        outer_cache[key]=(pp,ss,cc)
                    pp,ss,cc=outer_cache[key];append_result(rows,preds,name,rep,fold,te,y,ids,pp,ss,score,cc)
                pd.DataFrame(inner_rows).to_csv(OUT/f'nir_inner_r{rep}_f{fold}.csv',index=False)
                pd.DataFrame(rows).to_csv(OUT/'nir_folds.csv',index=False);pd.DataFrame(preds).to_csv(OUT/'nir_predictions.csv',index=False)
                splits.append(dict(repeat=rep,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),auxiliary_middle_ids=ids[mid].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]))
                (OUT/'nir_splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
                print(f'NIR repeat {rep+1} fold {fold+1}; elapsed {time.time()-start:.0f}s; '+', '.join(f'{r["model"]}={r["BA"]:.3f}' for r in rows[-6:]),flush=True)
        record_warnings(caught,'nir')
    print('NIR complete',flush=True)

RSCATTER=['area','snv','msc']
RSELECT=[dict(method='all',count=0,start_rank=0)]+[dict(method='SPA',count=n,start_rank=r) for n in [10,20] for r in [0,1,2]]
RCLASSIFIERS=[dict(type='linear_svm',C=c) for c in [.1,1]]+[dict(type='rbf_svm',C=c) for c in [1,10]]+[dict(type='pls_da',k=k) for k in [2,4]]

def read_raman():
    """Same 42 paired extremes, original labels and label-free per-spectrum preprocessing as the fusion run."""
    d=np.load(HERE/'fusion_comparison'/'paired_preprocessed.npz')
    lab=pd.read_csv(HERE/'fusion_comparison'/'paired_sample_labels.csv',dtype={'ID':str})
    ids=d['sample_ids'].astype(str);assert np.array_equal(ids,lab.ID.to_numpy())
    y=lab.label.to_numpy(dtype=int);assert np.bincount(y).tolist()==[29,13]
    grid=d['raman_uniform_wavenumbers'];recipes=json.loads((HERE/'fusion_comparison'/'protocol.json').read_text(encoding='utf8'))['raman']
    arrays=[d[f'Raman_{i}'] for i in range(len(recipes))]
    assert all(a.shape==(42,len(grid)) and np.isfinite(a).all() for a in arrays)
    return arrays,recipes,grid,y,ids

def raman_scatter(a,b,scatter):
    if scatter=='snv':
        f=lambda x:(x-x.mean(axis=1,keepdims=True))/x.std(axis=1,keepdims=True,ddof=1)
        return f(a),f(b)
    if scatter=='msc':
        a,b,_=msc_fit_transform(a,b);return a,b
    return a,b

def raman_classify(a,b,y,cfg):
    """Returns decision scores for b. Autoscaling is fitted on the training rows only."""
    sc=StandardScaler().fit(a);za,zb=sc.transform(a),sc.transform(b)
    if cfg['type']=='pls_da':
        return PLSRegression(n_components=cfg['k'],scale=False,max_iter=1000).fit(za,y).predict(zb).ravel()
    m=SVC(C=cfg['C'],kernel='linear' if cfg['type']=='linear_svm' else 'rbf',gamma='scale',class_weight='balanced').fit(za,y)
    return m.decision_function(zb)

def run_raman():
    arrays,recipes,grid,y,ids=read_raman();n=len(y)
    local=[dict(selector=s,classifier=c) for s in RSELECT for c in RCLASSIFIERS if not (c['type']=='pls_da' and s['method']=='SPA' and c['k']>s['count'])]
    protocol={'seeds':SEEDS,'outer_folds':5,'inner_folds':3,'samples':'42 paired extremes (29 low / 13 high), original 126-cohort labels, identical outer splits to fusion_comparison',
        'preprocessing':'reused fusion_comparison per-spectrum recipes (uniform grid; none/AsLS/airPLS lambda 1e4/1e6; SG9/poly2; area normalisation), then scatter area-only / SNV / MSC; MSC reference = mean of the current training rows only',
        'baselines':recipes,'scatter':RSCATTER,'selectors':RSELECT,'classifiers':RCLASSIFIERS,'candidate_count':len(arrays)*len(RSCATTER)*len(local),
        'SPA':'same spa_paths as NIR; start ranking by |corr| with training binary label; refitted inside every inner and outer training set',
        'SPA_counts':'10 and 20 only. Inner training sets hold ~22 samples, so at most ~21 linearly independent projections exist; 30 is infeasible (also exceeds outer-training rank ~32).',
        'SVM':'autoscaled selected variables, class_weight=balanced, decision_function score, fixed threshold 0',
        'PLS_DA':'autoscaled selected variables, PLS(scale=False) on 0/1 label; threshold chosen on inner OOF scores (max mean inner BA, ties nearest 0.5)'}
    (OUT/'raman_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    names=['Raman_area_full','Raman_SNV_full','Raman_MSC_full','Raman_scatter_selected','Raman_SPA_selected','Raman_joint_selected']
    rows=[];preds=[];splits=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for rep,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(arrays[0],y)):
                inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(arrays[0][tr],y[tr]))
                best={nm:(-np.inf,None) for nm in names};inner_rows=[]
                for bi,raw in enumerate(arrays):
                    for scatter in RSCATTER:
                        oof=np.zeros((len(local),len(tr)))
                        for it,iv in inner:
                            fit=tr[it];val=tr[iv];assert not set(fit)&set(np.r_[val,te])
                            a,b=raman_scatter(raw[fit],raw[val],scatter)
                            paths=spa_paths(a,y[fit].astype(float),max_count=20)
                            for j,cfg in enumerate(local):
                                s=cfg['selector'];chosen=np.arange(a.shape[1]) if s['method']=='all' else paths[s['start_rank']][:s['count']]
                                oof[j,iv]=raman_classify(a[:,chosen],b[:,chosen],y[fit],cfg['classifier'])
                        for j,cfg in enumerate(local):
                            if cfg['classifier']['type']=='pls_da':score,threshold=choose_threshold(oof[j],y[tr],inner,.5)
                            else:
                                threshold=0.0;score=float(np.mean([balanced_accuracy_score(y[tr][iv],(oof[j,iv]>=0).astype(int)) for _,iv in inner]))
                            config=dict(baseline=bi,baseline_cfg=recipes[bi],scatter=scatter,**cfg,threshold=threshold)
                            inner_rows.append(dict(inner_BA=score,config=json.dumps(config,sort_keys=True)))
                            if cfg['selector']['method']=='all':groups=[{'area':'Raman_area_full','snv':'Raman_SNV_full','msc':'Raman_MSC_full'}[scatter],'Raman_scatter_selected','Raman_joint_selected']
                            else:groups=['Raman_SPA_selected','Raman_joint_selected']
                            for g in groups:
                                if score>best[g][0]+1e-12:best[g]=(score,config)
                outer_cache={}
                for name,(score,cfg) in best.items():
                    key=json.dumps(cfg,sort_keys=True)
                    if key not in outer_cache:
                        a,b=raman_scatter(arrays[cfg['baseline']][tr],arrays[cfg['baseline']][te],cfg['scatter'])
                        sel=cfg['selector'];chosen=np.arange(a.shape[1]) if sel['method']=='all' else spa_paths(a,y[tr].astype(float),max_count=20)[sel['start_rank']][:sel['count']]
                        ss=raman_classify(a[:,chosen],b[:,chosen],y[tr],cfg['classifier']);pp=(ss>=cfg['threshold']).astype(int)
                        cc=dict(cfg)
                        if sel['method']=='SPA':cc['selected_wavenumbers']=grid[chosen].round(2).tolist()
                        outer_cache[key]=(pp,ss,cc)
                    pp,ss,cc=outer_cache[key];append_result(rows,preds,name,rep,fold,te,y,ids,pp,ss,score,cc)
                pd.DataFrame(inner_rows).to_csv(OUT/f'raman_inner_r{rep}_f{fold}.csv',index=False)
                pd.DataFrame(rows).to_csv(OUT/'raman_folds.csv',index=False);pd.DataFrame(preds).to_csv(OUT/'raman_predictions.csv',index=False)
                splits.append(dict(repeat=rep,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]))
                (OUT/'raman_splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
                print(f'Raman repeat {rep+1} fold {fold+1}; elapsed {time.time()-start:.0f}s; '+', '.join(f'{r["model"]}={r["BA"]:.3f}' for r in rows[-6:]),flush=True)
        record_warnings(caught,'raman')
    print('Raman complete',flush=True)

def record_warnings(caught,prefix):
    counts={}
    for w in caught:
        key=f'{w.category.__name__}: {w.message}';counts[key]=counts.get(key,0)+1
    (OUT/f'{prefix}_warnings.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
    assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)

def checks():
    rng=np.random.default_rng(42);v=rng.normal(size=80)
    a=np.array([(1+i*.1)*v+i for i in range(12)])
    b=np.array([2*v+1,3*v-2]);aa,bb,ref=msc_fit_transform(a,b)
    assert np.allclose(aa,np.broadcast_to(ref,aa.shape)) and np.allclose(bb,np.broadcast_to(ref,bb.shape))
    _,b2,ref2=msc_fit_transform(a,np.r_[b,[rng.normal(size=80)*1e5]])
    assert np.allclose(bb,b2[:2]) and np.array_equal(ref,ref2)
    z=rng.normal(size=(100,5));z[:,1]=z[:,0];target=z[:,0]+.1*rng.normal(size=100)
    for path in spa_paths(z,target,max_count=3):
        assert len(set(path))==3 and not {0,1}<=set(path)
    (OUT/'implementation_checks.json').write_text(json.dumps({'MSC_affine_scatter_recovery':True,'MSC_test_batch_independence':True,'SPA_avoids_exact_duplicate_predictors':True},indent=2),encoding='utf8')
    print('MSC and SPA implementation checks passed')

if __name__=='__main__':
    if sys.argv[1]=='nir':run_nir()
    elif sys.argv[1]=='raman':run_raman()
    elif sys.argv[1]=='checks':checks()
    else:raise SystemExit('Use nir, raman or checks.')
