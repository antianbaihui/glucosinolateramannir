"""Paired exploratory algorithm comparison on the same 88 extreme samples.

Run classical models: python compare_algorithms.py classical
Run local TabPFN v2: python compare_algorithms.py tabpfn
Summarize and audit: python report_algorithms.py
All cohort eligibility is fixed from the previous analysis; no independent holdout exists.
"""
import os,sys,json,time,warnings,hashlib,itertools
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'algorithm_comparison'; OUT.mkdir(exist_ok=True)
ASCII_DEPS=Path(os.environ.get('GLUCOSINOLATE_TABPFN_DEPS',r'C:\Users\22258006\.codex\visualizations\2026\09\28\01a0e676-9363-70a2-9c5d-8e7f7d545738\ml_deps'))
if len(sys.argv)>1 and sys.argv[1]=='tabpfn':
    sys.path.insert(0,str(ASCII_DEPS))
    sys.path.append(str(HERE/'deps'))  # xlrd for the supplied XLS reference table
else:
    sys.path.insert(0,str(HERE/'deps'))
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['TABPFN_DISABLE_TELEMETRY']='1'
os.environ['DO_NOT_TRACK']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TABPFN_MODEL_CACHE_DIR']=str(OUT/'models')
os.environ['HF_HOME']=str(OUT/'hf_cache')
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,matthews_corrcoef,confusion_matrix
from sklearn.exceptions import ConvergenceWarning

SEEDS=[42,2026,87]
BANDS={'full':(3999,10002),'original_4000_7500':(4000,7500),'continuous_4040_5810':(4040,5810)}
RECIPES=[(False,0),(True,0),(False,1),(True,1),(False,2),(True,2)]
MODEL_NAMES=['PLS_regression_all','shrinkage_LDA','ridge_logistic','elasticnet_logistic','selected_wavelength_SVM']

def data():
    chem=pd.read_excel(HERE.parent/'glucosinolate.xls')
    ids=chem.iloc[:,0].astype(str).str.replace(r'\.0$','',regex=True).str.strip()
    z=chem.iloc[:,1:].apply(pd.to_numeric,errors='raise')
    assert z.shape==(126,7) and np.isfinite(z.to_numpy()).all()
    totals=pd.Series(z.sum(axis=1).to_numpy(),index=ids)
    raw=pd.read_csv(HERE.parent/'TQData.csv',encoding='gbk')
    w=pd.to_numeric(raw.iloc[:,0],errors='coerce'); ok=w.notna();w=w[ok].to_numpy()
    frame=raw.loc[ok].iloc[:,1:].astype(float).T
    frame.index=frame.index.str.extract(r'^(\d+)',expand=False)
    assert frame.index.notna().all()
    assert pd.Series(frame.index).value_counts().eq(2).all()
    frame=frame.groupby(level=0).mean()
    # Preserve exact existing outer split row order; cross-check labels and chemistry.
    lab=pd.read_csv(HERE/'sample_labels.csv',dtype={'ID':str})
    extreme_ids=lab.ID.to_numpy(); middle_ids=np.array(sorted(set(frame.index)-set(extreme_ids)))
    allids=np.r_[extreme_ids,middle_ids]
    assert len(allids)==126 and len(set(allids))==126
    y=lab.label.to_numpy(dtype=int)
    t=totals.loc[allids].to_numpy()
    assert np.allclose(t[:88],lab.Total.to_numpy())
    order=np.argsort(w);w=w[order];x=frame.loc[allids].to_numpy()[:,order]
    assert np.isfinite(x).all() and np.isfinite(t).all()
    return x,w,y,t,allids

def cache_arrays(x,w):
    out={}
    for band,(lo,hi) in BANDS.items():
        a=x[:,(w>=lo)&(w<=hi)]
        for snv,der in RECIPES:
            b=a.copy()
            if snv:b=(b-b.mean(axis=1,keepdims=True))/b.std(axis=1,keepdims=True,ddof=1)
            out[(band,snv,der)]=savgol_filter(b,15,2,deriv=der,delta=float(np.median(np.diff(w))),axis=1)
    return out

def metrics(y,p,s):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    return dict(BA=balanced_accuracy_score(y,p),AUC=roc_auc_score(y,s),MCC=matthews_corrcoef(y,p),
        sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/max(tp+fp,1),
        selected_fraction=float(np.mean(p)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def json_cfg(c):return json.dumps(c,ensure_ascii=False,sort_keys=True)

def representation(a,b,y,kind):
    if kind=='PLS':
        m=PLSRegression(n_components=8,scale=True,max_iter=1000).fit(a,y)
        return m.transform(a),m.transform(b)
    scale=StandardScaler().fit(a)
    m=PCA(n_components=20,svd_solver='full').fit(scale.transform(a))
    return m.transform(scale.transform(a)),m.transform(scale.transform(b))

def linear_configs():
    cs=[]
    for kind,ks in [('PLS',[3,5,8]),('PCA',[5,10,20])]:
        for k in ks:
            for shrink in ['auto',0.1,0.5]:cs.append(('shrinkage_LDA',dict(kind=kind,k=k,shrinkage=shrink)))
            for c in [.01,.1,1,10]:cs.append(('ridge_logistic',dict(kind=kind,k=k,C=c)))
            for c in [.1,1,10]:
                for ratio in [.2,.8]:cs.append(('elasticnet_logistic',dict(kind=kind,k=k,C=c,l1_ratio=ratio)))
    return cs
LINEAR=linear_configs()
WAVE=[dict(k=k,kernel=kernel,C=c) for k in [10,30,80] for kernel in ['linear','rbf'] for c in [.1,1,10]]
REG=[dict(k=k,log=log) for k in [2,5,8,12] for log in [False,True]]

def classifier(name,cfg):
    if name=='shrinkage_LDA':return LinearDiscriminantAnalysis(solver='lsqr',shrinkage=cfg['shrinkage'])
    if name=='ridge_logistic':return LogisticRegression(C=cfg['C'],solver='lbfgs',max_iter=5000,class_weight='balanced')
    return LogisticRegression(C=cfg['C'],penalty='elasticnet',l1_ratio=cfg['l1_ratio'],solver='saga',max_iter=10000,tol=1e-4,random_state=42,class_weight='balanced')

def linear_prediction(name,cfg,zt,zv,y):
    k=cfg['k'];sc=StandardScaler().fit(zt[:,:k]);a=sc.transform(zt[:,:k]);b=sc.transform(zv[:,:k])
    model=classifier(name,cfg).fit(a,y)
    return model.predict(b),model.decision_function(b)

def wave_prediction(cfg,a,b,y):
    f,_=f_classif(a,y)
    chosen=np.argsort(-np.nan_to_num(f,nan=-np.inf),kind='stable')[:cfg['k']]
    sc=StandardScaler().fit(a[:,chosen]);aa=sc.transform(a[:,chosen]);bb=sc.transform(b[:,chosen])
    model=SVC(C=cfg['C'],kernel=cfg['kernel'],gamma='scale',class_weight='balanced').fit(aa,y)
    return model.predict(bb),model.decision_function(bb),chosen

def reg_prediction(cfg,a,b,t):
    target=np.log1p(t) if cfg['log'] else t
    m=PLSRegression(n_components=cfg['k'],scale=True,max_iter=1000).fit(a,target)
    s=m.predict(b).ravel()
    return np.expm1(s) if cfg['log'] else s

def add_result(rows,preds,name,rep,fold,tr,te,y,ids,p,s,inner,cfg):
    rows.append(dict(model=name,repeat=rep,fold=fold,inner_BA=float(inner),config=json_cfg(cfg),**metrics(y[te],p,s)))
    preds.extend(dict(model=name,repeat=rep,fold=fold,ID=str(ids[i]),y=int(y[i]),prediction=int(v),score=float(q)) for i,v,q in zip(te,p,s))

def write_results(rows,preds,prefix):
    pd.DataFrame(rows).to_csv(OUT/f'{prefix}_folds.csv',index=False)
    pd.DataFrame(preds).to_csv(OUT/f'{prefix}_predictions.csv',index=False)

def classical():
    x,w,y,t,ids=data();arr=cache_arrays(x,w);mid=np.arange(88,126)
    rows=[];preds=[];start=time.time();checks=[]
    protocol={'seeds':SEEDS,'outer':5,'inner':4,'bands':BANDS,'recipes':RECIPES,'linear':LINEAR,'wavelength_SVM':WAVE,'regression':REG,
        'regression_threshold':'inner OOF prediction midpoints; optimize mean inner-fold BA; ties closest to midpoint of outer-training class means',
        'regression_middle_samples':'all 38 middle samples are auxiliary training only in each inner/outer regression fit; never in classification metric',
        'tabpfn':'v2, CPU, 2 estimators, 3 bands x SNV derivatives 0/1 x PCA10/PLS5; 4-fold inner and same repeated outer',
        'eligibility':'fixed extremes of previously inspected cohort; exploratory only; IDs may not represent independent genotypes'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    # Do not silence convergence problems; record every warning for audit.
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter('always')
        for rep,seed in enumerate(SEEDS):
            for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x[:88],y)):
                inner=list(StratifiedKFold(4,shuffle=True,random_state=seed+fold+1).split(x[tr],y[tr]))
                best={n:(-np.inf,None) for n in MODEL_NAMES}
                all_inner=[]
                for (band,snv,der),a in arr.items():
                    lscore=np.zeros(len(LINEAR));wscore=np.zeros(len(WAVE));regpred=np.zeros((len(REG),len(tr)))
                    for it,iv in inner:
                        itg,ivg=tr[it],tr[iv]
                        assert not set(itg)&set(te) and not set(ivg)&set(te)
                        zs={kind:representation(a[itg],a[ivg],y[itg],kind) for kind in ['PLS','PCA']}
                        for j,(name,cfg) in enumerate(LINEAR):
                            p,_=linear_prediction(name,cfg,*zs[cfg['kind']],y[itg])
                            lscore[j]+=balanced_accuracy_score(y[ivg],p)/4
                        for j,cfg in enumerate(WAVE):
                            p,_,_=wave_prediction(cfg,a[itg],a[ivg],y[itg])
                            wscore[j]+=balanced_accuracy_score(y[ivg],p)/4
                        train_reg=np.r_[itg,mid]
                        assert not set(train_reg)&set(np.r_[ivg,te])
                        for j,cfg in enumerate(REG):
                            regpred[j,iv]=reg_prediction(cfg,a[train_reg],a[ivg],t[train_reg])
                    for j,(name,cfg) in enumerate(LINEAR):
                        cf=dict(cfg,band=band,snv=snv,derivative=der)
                        all_inner.append(dict(model=name,inner_BA=float(lscore[j]),config=json_cfg(cf)))
                        if lscore[j]>best[name][0]+1e-12:best[name]=(lscore[j],cf)
                    for j,cfg in enumerate(WAVE):
                        cf=dict(cfg,band=band,snv=snv,derivative=der)
                        all_inner.append(dict(model='selected_wavelength_SVM',inner_BA=float(wscore[j]),config=json_cfg(cf)))
                        if wscore[j]>best['selected_wavelength_SVM'][0]+1e-12:best['selected_wavelength_SVM']=(wscore[j],cf)
                    target_midpoint=(t[tr[y[tr]==0]].mean()+t[tr[y[tr]==1]].mean())/2
                    for j,cfg in enumerate(REG):
                        u=np.unique(regpred[j]);thresholds=np.r_[u[0]-1,(u[:-1]+u[1:])/2,u[-1]+1]
                        scores=np.zeros(len(thresholds))
                        for _,iv in inner:
                            yp=regpred[j,iv,None]>=thresholds[None,:]
                            yy=y[tr[iv]]
                            scores+=(yp[yy==1].mean(axis=0)+(~yp[yy==0]).mean(axis=0))/8
                        winners=np.flatnonzero(scores>=scores.max()-1e-12)
                        k=winners[np.argmin(abs(thresholds[winners]-target_midpoint))]
                        cf=dict(cfg,band=band,snv=snv,derivative=der,threshold=float(thresholds[k]))
                        all_inner.append(dict(model='PLS_regression_all',inner_BA=float(scores[k]),config=json_cfg(cf)))
                        if scores[k]>best['PLS_regression_all'][0]+1e-12:best['PLS_regression_all']=(scores[k],cf)
                pd.DataFrame(all_inner).to_csv(OUT/f'inner_scores_r{rep}_f{fold}.csv',index=False)
                for name,(score,cfg) in best.items():
                    a=arr[(cfg['band'],cfg['snv'],cfg['derivative'])]
                    if name=='PLS_regression_all':
                        train_reg=np.r_[tr,mid];s=reg_prediction(cfg,a[train_reg],a[te],t[train_reg]);p=(s>=cfg['threshold']).astype(int)
                    elif name=='selected_wavelength_SVM':
                        p,s,chosen=wave_prediction(cfg,a[tr],a[te],y[tr]);lo,hi=BANDS[cfg['band']]
                        cfg=dict(cfg,selected_wavenumbers=w[(w>=lo)&(w<=hi)][chosen].tolist())
                    else:
                        zt,zv=representation(a[tr],a[te],y[tr],cfg['kind']);p,s=linear_prediction(name,cfg,zt,zv,y[tr])
                    add_result(rows,preds,name,rep,fold,tr,te,y,ids,p,s,score,cfg)
                checks.append(dict(repeat=rep,fold=fold,test_ids=ids[te].tolist(),train_extreme_ids=ids[tr].tolist(),regression_auxiliary_ids=ids[mid].tolist()))
                write_results(rows,preds,'classical')
                (OUT/'split_audit.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
                print(f'classical repeat {rep+1} fold {fold+1}; elapsed {time.time()-start:.0f}s; '+', '.join(f'{r["model"]}={r["BA"]:.3f}' for r in rows[-5:]),flush=True)
        counts={}
        for wr in warns:
            key=f'{wr.category.__name__}: {wr.message}'
            counts[key]=counts.get(key,0)+1
        (OUT/'warnings.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
        assert not any(issubclass(wr.category,ConvergenceWarning) for wr in warns),'Convergence warning; inspect before interpreting.'
    print('Classical comparison complete.',flush=True)

def tabpfn():
    from tabpfn import TabPFNClassifier
    import torch
    torch.set_num_threads(2)
    x,w,y,t,ids=data();arr=cache_arrays(x,w);rows=[];preds=[];start=time.time()
    cfgs=[dict(band=b,snv=True,derivative=d,kind=kind,k=k) for b in BANDS for d in [0,1] for kind,k in [('PCA',10),('PLS',5)]]
    def fit(cfg,a,tr,te,seed):
        zt,zv=representation(a[tr],a[te],y[tr],cfg['kind'])
        m=TabPFNClassifier(n_estimators=2,device='cpu',random_state=seed,n_jobs=1,
            model_path=str(OUT/'models'/'tabpfn-v2-classifier.ckpt'))
        m.fit(zt[:,:cfg['k']],y[tr]);s=m.predict_proba(zv[:,:cfg['k']])[:,1]
        return (s>=.5).astype(int),s
    for rep,seed in enumerate(SEEDS):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x[:88],y)):
            inner=list(StratifiedKFold(4,shuffle=True,random_state=seed+fold+1).split(x[tr],y[tr]))
            scores=[]
            for cfg in cfgs:
                a=arr[(cfg['band'],cfg['snv'],cfg['derivative'])];score=0
                for it,iv in inner:
                    p,_=fit(cfg,a,tr[it],tr[iv],seed+fold);score+=balanced_accuracy_score(y[tr[iv]],p)/4
                scores.append(score)
            winner=int(np.argmax(scores));cfg=cfgs[winner];a=arr[(cfg['band'],cfg['snv'],cfg['derivative'])]
            p,s=fit(cfg,a,tr,te,seed+fold)
            add_result(rows,preds,'TabPFN_v2',rep,fold,tr,te,y,ids,p,s,scores[winner],cfg)
            pd.DataFrame([dict(config=json_cfg(c),inner_BA=v) for c,v in zip(cfgs,scores)]).to_csv(OUT/f'tabpfn_inner_r{rep}_f{fold}.csv',index=False)
            write_results(rows,preds,'tabpfn')
            print(f'TabPFN repeat {rep+1} fold {fold+1}; BA={rows[-1]["BA"]:.3f}; elapsed {time.time()-start:.0f}s',flush=True)
    import sklearn,tabpfn as tab,scipy
    (OUT/'tabpfn_versions.json').write_text(json.dumps({'torch':torch.__version__,'sklearn':sklearn.__version__,'tabpfn':getattr(tab,'__version__','2.0.9'),'scipy':scipy.__version__,'checkpoint_sha256':hashlib.sha256((OUT/'models'/'tabpfn-v2-classifier.ckpt').read_bytes()).hexdigest()},indent=2),encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='classical':classical()
    elif sys.argv[1]=='tabpfn':tabpfn()
    else:raise SystemExit('Use classical or tabpfn; summary script is separate.')
