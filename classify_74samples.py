"""Reproducible, bounded all-family comparison for exactly the submitted cohort.

Usage: classify_74samples.py audit|classical|deep|tabpfn|report [nir|nir_paired|raman|fusion]
The cohort was selected after earlier OOF errors; all performance is exploratory.
"""
import os, sys, json, time, hashlib, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / 'classification_74samples'
OUT.mkdir(exist_ok=True)
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
ML_DEPS = Path(r'C:\Users\22258006\.codex\visualizations\2026\09\28\01a0e676-9363-70a2-9c5d-8e7f7d545738\ml_deps')
MODE = sys.argv[1] if len(sys.argv) > 1 else 'audit'
if MODE == 'tabpfn':
    sys.path.insert(0, str(ML_DEPS))
    import numpy, pandas, sklearn, scipy
    from tabpfn import TabPFNClassifier
import all_algorithm_suite as base
from all_algorithm_suite import np, pd, StandardScaler, PLSRegression, StratifiedKFold, SVC
from optimize_single_modalities import spa_paths, reg_predict, choose_threshold, append_result, metric
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning
SEEDS = [42, 2026, 87]
COHORTS = ['nir', 'nir_paired', 'raman', 'fusion']

def submitted():
    a = pd.read_csv(HERE.parent/'74samples.csv', dtype={'ID':str})
    assert list(a.columns) == ['ID','Total','label']
    assert len(a) == 74 and a.ID.nunique() == 74
    assert set(a.label) == {0,1} and np.isfinite(a.Total).all()
    return a

def inputs(cohort):
    a = submitted()
    raw, w, y, ids = base.inputs(cohort)
    # Preserve submitted order; paired cohorts are its exact Raman intersection.
    ids = ids.astype(str)
    wanted = a[a.ID.isin(ids)].copy()
    ix = np.array([np.flatnonzero(ids == sid)[0] for sid in wanted.ID])
    assert np.array_equal(y[ix], wanted.label.to_numpy())
    assert len(ix) == (74 if cohort == 'nir' else 37)
    return [r[ix] for r in raw], w, y[ix], ids[ix]

def audit():
    from compare_algorithms import data
    a = submitted(); x,w,y,t,ids = data()
    ix = np.array([np.flatnonzero(ids == sid)[0] for sid in a.ID])
    assert (ix < 88).all() and np.array_equal(y[ix], a.label)
    assert np.allclose(t[ix], a.Total, rtol=0, atol=1e-9)
    previous = pd.read_csv(HERE/'exclusion_exploration/保留74份分类样品.csv',dtype={'ID':str})
    sources = [HERE.parent/'74samples.csv', HERE.parent/'TQData.csv', HERE.parent/'glucosinolate.xls', HERE/'fusion_comparison/paired_preprocessed.npz']
    info = dict(input_rows=74, low=int((a.label==0).sum()), high=int((a.label==1).sum()),
        exact_same_ids_as_previous_error_based_exclusion=set(a.ID)==set(previous.ID),
        totals_and_labels_match=True, auxiliary_samples=0,
        input_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cohort_details={}, limitation='Same 74 IDs as previous exclusion of 14 samples wrong in all three OOF repeats. Post-hoc selection bias remains despite nested CV. No further sample removal.')
    for cohort in COHORTS:
        raw,ww,yy,ii=inputs(cohort)
        assert all(np.isfinite(r).all() for r in raw)
        info['cohort_details'][cohort]=dict(n=len(ii),low=int((yy==0).sum()),high=int((yy==1).sum()),features=[r.shape[1] for r in raw])
        a.set_index('ID').loc[ii].reset_index().to_csv(OUT/f'{cohort}_samples.csv', index=False, encoding='utf-8-sig')
    (OUT/'audit.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(info,ensure_ascii=True,indent=2),flush=True)

def configurations(cohort):
    cfg = base.configurations(cohort)
    cfg['Linear_SVM']=[dict(C=c) for c in [.1,1,10]]
    cfg['shrinkage_LDA']=[dict(rep=r,shrinkage=h) for r in ['PCA','PLS'] for h in ['auto',.1,.5]]
    cfg['ridge_logistic']=[dict(rep=r,C=c) for r in ['PCA','PLS'] for c in [.01,.1,1,10]]
    cfg['elasticnet_logistic']=[dict(rep=r,C=c,l1_ratio=l) for r in ['PCA','PLS'] for c in [.1,1,10] for l in [.2,.8]]
    cfg['F_selected_SVM']=[dict(count=n,kernel=k,C=c) for n in [10,30,80] for k in ['linear','rbf'] for c in [.1,1,10]]
    cfg['PLS_regression']=[dict(k=k,log=l) for k in [2,5,8] for l in [False,True]]
    counts=[10,20,30] if cohort=='nir' else [10]
    cfg['SPA_PLS_regression']=[dict(count=n,start=r,k=k,log=l) for n in counts for r in [0,1,2] for k in [2,5] for l in [False,True]]
    if cohort=='fusion':
        cfg['Midlevel_PLS_SVM']=[dict(k=k,kernel=z,C=c) for k in [2,4] for z in ['linear','rbf'] for c in [.1,1,10]]
    return cfg

def representations(blocks,y,total,seed,cohort):
    rep=base.representations(blocks,y,seed)
    a,b=rep['full'];f,_=f_classif(a,y)
    rep['f_order']=np.argsort(-np.nan_to_num(f,nan=-np.inf),kind='stable')
    rep['spa_paths']=spa_paths(a,total,30 if cohort=='nir' else 10)
    rep['total']=total
    if cohort=='fusion':
        for k in [2,4]:
            values=[]
            for aa,bb in blocks:
                m=PLSRegression(n_components=k,scale=False,max_iter=1000).fit(aa,y)
                za,zb=m.transform(aa),m.transform(bb);ss=StandardScaler().fit(za)
                values.append((ss.transform(za)/np.sqrt(k),ss.transform(zb)/np.sqrt(k)))
            rep[f'mid{k}']=tuple(np.concatenate([v[j] for v in values],axis=1) for j in [0,1])
    return rep

def predict(name,cfg,reps,y,seed):
    if name=='PCA_KNN':
        # Exact tree search avoids a Windows threadpool DLL path issue in brute search.
        a,b=reps['PCA']
        model=base.KNeighborsClassifier(n_neighbors=cfg['k'],weights=cfg['weights'],algorithm='kd_tree').fit(a,y)
        s=model.predict_proba(b)[:,1]
        return (s>=.5).astype(int),s
    if name in ['PLS_regression','SPA_PLS_regression']:
        a,b=reps['full']
        if name.startswith('SPA'):
            cols=reps['spa_paths'][cfg['start']][:cfg['count']];a=a[:,cols];b=b[:,cols]
        score=reg_predict(a,b,reps['total'],cfg)
        return None,score
    if name=='Linear_SVM':
        a,b=reps['PLS'];model=SVC(kernel='linear',C=cfg['C'],class_weight='balanced')
    elif name=='F_selected_SVM':
        cols=reps['f_order'][:cfg['count']];a,b=reps['full'];sc=StandardScaler().fit(a[:,cols]);a,b=sc.transform(a[:,cols]),sc.transform(b[:,cols])
        model=SVC(kernel=cfg['kernel'],C=cfg['C'],gamma='scale',class_weight='balanced')
    elif name=='Midlevel_PLS_SVM':
        a,b=reps[f'mid{cfg["k"]}'];model=SVC(kernel=cfg['kernel'],C=cfg['C'],gamma='scale',class_weight='balanced')
    elif name in ['shrinkage_LDA','ridge_logistic','elasticnet_logistic']:
        a,b=reps[cfg['rep']];sc=StandardScaler().fit(a);a,b=sc.transform(a),sc.transform(b)
        if name=='shrinkage_LDA':model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage=cfg['shrinkage'])
        elif name=='ridge_logistic':model=LogisticRegression(C=cfg['C'],solver='lbfgs',max_iter=10000,class_weight='balanced')
        else:model=LogisticRegression(C=cfg['C'],penalty='elasticnet',l1_ratio=cfg['l1_ratio'],solver='saga',max_iter=30000,tol=1e-4,random_state=seed,class_weight='balanced')
    else:return base.predict(name,cfg,reps,y,seed)
    model.fit(a,y);return model.predict(b),model.decision_function(b)

def classical(cohort):
    folder=OUT/cohort;folder.mkdir(exist_ok=True)
    raw,w,y,ids=inputs(cohort);total=submitted().set_index('ID').loc[ids,'Total'].to_numpy()
    cfgs=configurations(cohort);innern=4 if cohort=='nir' else 3
    protocol=dict(cohort=cohort,n=len(ids),seeds=SEEDS,outer=5,inner=innern,auxiliary_samples=0,
        configurations=cfgs,recipes=['NIR4000-7500 SNV SG1 / Raman airPLS1e6 SG0 area','NIR4000-7500 MSC SG2 / Raman airPLS1e6 SG1 area'],
        threshold='Regression threshold selected from inner OOF predictions only; no total concentration used as input feature',
        scope='All previously listed applicable families, bounded grids; not exhaustive hyperparameters or wavelength search',
        limitation='User list equals previous error-based exclusion cohort. Exploratory conditional performance only.')
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[];preds=[];splits=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for repeat,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(raw[0],y)):
                inner=list(StratifiedKFold(innern,shuffle=True,random_state=seed+fold+1).split(raw[0][tr],y[tr]))
                best={name:(-np.inf,None) for name in cfgs};inner_records=[]
                midpoint=(total[tr[y[tr]==0]].mean()+total[tr[y[tr]==1]].mean())/2
                for recipe in [0,1]:
                    scores={name:np.zeros(len(cs)) for name,cs in cfgs.items()}
                    ro={name:np.zeros((len(cfgs[name]),len(tr))) for name in ['PLS_regression','SPA_PLS_regression']}
                    for ii,(it,iv) in enumerate(inner):
                        fit=tr[it];val=tr[iv]
                        assert not set(fit)&set(np.r_[val,te])
                        reps=representations(base.prep_blocks(raw,w,fit,val,recipe,cohort),y[fit],total[fit],seed+100*fold+ii,cohort)
                        for name,cs in cfgs.items():
                            for j,cfg in enumerate(cs):
                                pp,ss=predict(name,cfg,reps,y[fit],seed+fold)
                                if name in ro:ro[name][j,iv]=ss
                                else:scores[name][j]+=base.balanced_accuracy_score(y[val],pp)/innern
                    for name,cs in cfgs.items():
                        for j,cfg in enumerate(cs):
                            conf=dict(recipe=recipe,parameters=cfg.copy())
                            if name in ro:
                                score,threshold=choose_threshold(ro[name][j],y[tr],inner,midpoint);conf['threshold']=threshold
                            else:score=float(scores[name][j])
                            inner_records.append(dict(model=name,inner_BA=score,config=json.dumps(conf,sort_keys=True)))
                            if score>best[name][0]+1e-12:best[name]=(score,conf)
                cache={};current={};currentpred={}
                for name,(score,cfg) in best.items():
                    r=cfg['recipe']
                    if r not in cache:cache[r]=representations(base.prep_blocks(raw,w,tr,te,r,cohort),y[tr],total[tr],seed+100*fold+99,cohort)
                    reps=cache[r];pp,ss=predict(name,cfg['parameters'],reps,y[tr],seed+fold)
                    if pp is None:pp=(ss>=cfg['threshold']).astype(int)
                    assert np.isfinite(ss).all()
                    cc=dict(cfg)
                    if name.startswith(('CARS','VIP')):cc['selected_indices']=reps['metadata'][('CARS' if name.startswith('CARS') else 'VIP')+'_indices']
                    if name=='F_selected_SVM':cc['selected_indices']=reps['f_order'][:cfg['parameters']['count']].tolist()
                    if name=='SPA_PLS_regression':cc['selected_indices']=reps['spa_paths'][cfg['parameters']['start']][:cfg['parameters']['count']].tolist()
                    current[name]=append_result(rows,preds,name,repeat,fold,te,y,ids,pp,ss,score,cc)
                    currentpred[name]=(pp,ss)
                winner=max(best,key=lambda name:best[name][0]);pp,ss=currentpred[winner]
                append_result(rows,preds,'nested_classical_selection',repeat,fold,te,y,ids,pp,ss,best[winner][0],dict(selected_family=winner,**best[winner][1]))
                pd.DataFrame(inner_records).to_csv(folder/f'inner_r{repeat}_f{fold}.csv',index=False)
                pd.DataFrame(rows).to_csv(folder/'fold_metrics.csv',index=False)
                pd.DataFrame(preds).to_csv(folder/'oof_predictions.csv',index=False)
                splits.append(dict(repeat=repeat,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]))
                (folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
                print(f'classical {cohort} {repeat+1}/{fold+1}; {time.time()-start:.0f}s; selected {winner}',flush=True)
        counts={}
        for v in caught:
            key=f'{v.category.__name__}: {v.message}';counts[key]=counts.get(key,0)+1
        (folder/'warnings.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
        assert not any(issubclass(v.category,ConvergenceWarning) for v in caught), 'Convergence warning: inspect before reporting'
    print('COMPLETE classical '+cohort,flush=True)

def deep(cohort):
    import deep_algorithm_suite as d
    d.OUT=OUT;d.inputs=inputs
    d.main(cohort)

def tabpfn(cohort):
    import torch
    torch.set_num_threads(1)
    folder=OUT/(cohort+'_tabpfn');folder.mkdir(exist_ok=True)
    raw,w,y,ids=inputs(cohort);innern=4 if cohort=='nir' else 3
    checkpoint=HERE/'algorithm_comparison/models/tabpfn-v2-classifier.ckpt'
    assert checkpoint.exists()
    cfgs=[dict(recipe=r,rep=kind) for r in [0,1] for kind in ['PCA','PLS']]
    protocol=dict(cohort=cohort,n=len(ids),seeds=SEEDS,outer=5,inner=innern,configurations=cfgs,n_estimators=2,device='cpu',auxiliary_samples=0,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),sklearn=sklearn.__version__,torch=torch.__version__)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    def fit(cfg,tr,te,seed):
        blocks=base.prep_blocks(raw,w,tr,te,cfg['recipe'],cohort);a,b=base.block_concat(blocks)
        if cfg['rep']=='PCA':
            ss=StandardScaler().fit(a);a,b=ss.transform(a),ss.transform(b);m=base.PCA(n_components=10,svd_solver='full').fit(a)
        else:m=PLSRegression(n_components=4,scale=False,max_iter=1000).fit(a,y[tr])
        a,b=m.transform(a),m.transform(b)
        model=TabPFNClassifier(n_estimators=2,device='cpu',random_state=seed,n_jobs=1,model_path=str(checkpoint))
        model.fit(a,y[tr]);s=model.predict_proba(b)[:,1]
        return (s>=.5).astype(int),s
    rows=[];preds=[];splits=[];start=time.time()
    for repeat,seed in enumerate(SEEDS):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(raw[0],y)):
            inner=list(StratifiedKFold(innern,shuffle=True,random_state=seed+fold+1).split(raw[0][tr],y[tr]));scores=[]
            for cfg in cfgs:
                value=0
                for it,iv in inner:
                    assert not set(tr[it])&set(np.r_[tr[iv],te])
                    pp,_=fit(cfg,tr[it],tr[iv],seed+fold);value+=base.balanced_accuracy_score(y[tr[iv]],pp)/innern
                scores.append(value)
            j=int(np.argmax(scores));p,s=fit(cfgs[j],tr,te,seed+fold)
            append_result(rows,preds,'TabPFN_v2',repeat,fold,te,y,ids,p,s,scores[j],cfgs[j])
            pd.DataFrame(rows).to_csv(folder/'fold_metrics.csv',index=False);pd.DataFrame(preds).to_csv(folder/'oof_predictions.csv',index=False)
            pd.DataFrame([dict(model='TabPFN_v2',inner_BA=v,config=json.dumps(c,sort_keys=True)) for c,v in zip(cfgs,scores)]).to_csv(folder/f'inner_r{repeat}_f{fold}.csv',index=False)
            splits.append(dict(repeat=repeat,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]))
            (folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
            print(f'TabPFN {cohort} {repeat+1}/{fold+1}; {time.time()-start:.0f}s',flush=True)
    print('COMPLETE TabPFN '+cohort,flush=True)

if __name__=='__main__':
    if MODE=='audit':audit()
    elif MODE in ['classical','deep','tabpfn']:
        cohorts=COHORTS[1:] if sys.argv[2]=='paired' else [sys.argv[2]]
        for cohort in cohorts:
            assert cohort in COHORTS
            globals()[MODE](cohort)
    else:raise SystemExit('Use audit, classical, deep or tabpfn')
