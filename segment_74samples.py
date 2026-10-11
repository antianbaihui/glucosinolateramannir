"""Paper-derived interval comparison on the submitted 74 NIR samples.

Run classical or ann. Fixed 17 candidate regions, identical previous nested splits.
No new sample filtering, no auxiliary samples, no outer-test-driven interval choice.
"""
import sys, os, json, time, warnings, hashlib
from pathlib import Path
import classify_74samples as old
from classify_74samples import np,pd,StandardScaler,PLSRegression
from optimize_single_modalities import nir_preprocess,reg_predict,choose_threshold,append_result
from compare_algorithms import data
HERE=Path(__file__).resolve().parent
OUT=HERE/'segmentation_74samples';OUT.mkdir(exist_ok=True)
PAPER=Path(r'C:\Users\22258006\Downloads\Yijun zhang glucosinolate.pdf')
BANDS={'A':(9500,9970),'B':(8400,8920),'C':(7850,8270),'D':(6110,7200),'E':(5020,5810),'F':(4040,4790)}
CLASSICAL=['Linear_SVM','SVM_RBF','ridge_logistic','shrinkage_LDA','PLS_regression']
CFG={name:old.configurations('nir')[name] for name in CLASSICAL}

def regions():
    rr=[dict(name='baseline_4000_7500',group='reference',ranges=[(4000,7500)]),
        dict(name='full',group='reference',ranges=[(3999,10002)]),
        dict(name='continuous_4040_5810',group='reference',ranges=[(4040,5810)])]
    rr.extend(dict(name=name,group='paper_single',ranges=[bounds]) for name,bounds in BANDS.items())
    for name in ['ABC','ABCD','ABCDE','ABCDEF']:
        rr.append(dict(name=name,group='paper_combination',ranges=[BANDS[c] for c in name]))
    for name in ['DE','DF','EF','DEF']:
        rr.append(dict(name=name,group='additional_combination',ranges=[BANDS[c] for c in name]))
    for r in rr:r['ranges']=sorted(r['ranges'])
    return rr

def inputs():
    x,w,y,t,ids=data();a=old.submitted()
    ix=np.array([np.flatnonzero(ids==sid)[0] for sid in a.ID])
    assert np.array_equal(y[ix],a.label) and np.allclose(t[ix],a.Total,rtol=0,atol=1e-9)
    return x[ix],w,y[ix],t[ix],ids[ix]

def prepare(x,w,tr,te,region,recipe):
    aa=[];bb=[]
    for lo,hi in region['ranges']:
        mask=(w>=lo)&(w<=hi);assert mask.sum()>=15
        delta=float(np.median(np.diff(w[mask])))
        a,b=nir_preprocess(x[tr][:,mask],x[te][:,mask],'snv' if recipe==0 else 'msc',1 if recipe==0 else 2,delta)
        aa.append(a);bb.append(b)
    a,b=np.concatenate(aa,axis=1),np.concatenate(bb,axis=1)
    sc=StandardScaler().fit(a)
    return [(sc.transform(a),sc.transform(b))]

def representations(blocks,y,total):
    a,b=old.base.block_concat(blocks)
    sc=StandardScaler().fit(a);za,zb=sc.transform(a),sc.transform(b)
    pca=old.base.PCA(n_components=10,svd_solver='full').fit(za)
    pls=PLSRegression(n_components=4,scale=False,max_iter=1000).fit(a,y)
    ta,tb=pls.transform(a),pls.transform(b);ss=StandardScaler().fit(ta)
    return dict(full=(a,b),PCA=(pca.transform(za),pca.transform(zb)),PLS=(ss.transform(ta),ss.transform(tb)),total=total)

def save(folder,rows,preds,records,rep,fold,region):
    pd.DataFrame(records).to_csv(folder/f'inner_r{rep}_f{fold}_{region}.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'fold_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(folder/'oof_predictions.csv',index=False)

def main(mode):
    assert mode in ['classical','ann']
    folder=OUT/mode;folder.mkdir(exist_ok=True)
    if mode=='ann':
        import deep_algorithm_suite as deep
        cfgs={'ANN':deep.CONFIG}
    else:cfgs=CFG
    x,w,y,total,ids=inputs();rr=regions();assert len(rr)==17
    splits=json.loads((old.OUT/'nir/splits.json').read_text(encoding='utf8'))
    lookup={sid:i for i,sid in enumerate(ids)}
    sourcefiles=[HERE.parent/'74samples.csv',HERE.parent/'TQData.csv',HERE.parent/'glucosinolate.xls']
    protocol=dict(mode=mode,n=74,low=36,high=38,auxiliary_samples=0,regions=rr,configs=cfgs,
        preprocessing=['SNV then SG(15,2) first derivative','training MSC then SG(15,2) second derivative'],
        segment_handling='Each disjoint contiguous band receives its own scatter correction and derivative. Concatenate in ascending wavenumber order; no interpolation or derivative over gaps.',
        representation='Identical previous PCA10/PLS4 and model grids, including ANN 60 epochs AdamW width16/lr.001 or width32/lr.003',
        cv='Exact previous 3x5 outer and4 inner ID splits. Fixed-region scores are post-hoc descriptions; region selection assessed separately using inner scores only.',
        paper='Zhang et al., Food Chemistry X 28 (2025) 102528, Fig3/page6 and Table2/page7. Pigment regression, not glucosinolate classification. Transfer regions only; use existing SG, not claim exact Norris derivative replication.',
        selection_bias='74 IDs equal previous error-based exclusion cohort; not unbiased external validation',
        source_sha256={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},
        paper_sha256=hashlib.sha256(PAPER.read_bytes()).hexdigest())
    (folder/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
    rows=[];preds=[];start=time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for split in splits:
            rep,fold=split['repeat'],split['fold'];seed=old.SEEDS[rep]
            tr=np.array([lookup[s] for s in split['train_ids']]);te=np.array([lookup[s] for s in split['test_ids']])
            inner=[(np.array([lookup[s] for s in v['train_ids']]),np.array([lookup[s] for s in v['validation_ids']])) for v in split['inner']]
            loc={int(v):i for i,v in enumerate(tr)}
            innerlocal=[(np.array([loc[int(v)] for v in it]),np.array([loc[int(v)] for v in iv])) for it,iv in inner]
            midpoint=(total[tr[y[tr]==0]].mean()+total[tr[y[tr]==1]].mean())/2
            for region in rr:
                records=[];best={n:(-np.inf,None) for n in cfgs}
                for recipe in [0,1]:
                    scores={n:np.zeros(len(cs)) for n,cs in cfgs.items()}
                    regoof=np.zeros((len(cfgs.get('PLS_regression',[])),len(tr)))
                    for ii,(fit,val) in enumerate(inner):
                        assert not set(fit)&set(np.r_[val,te])
                        blocks=prepare(x,w,fit,val,region,recipe)
                        features=deep.deep_inputs(blocks) if mode=='ann' else representations(blocks,y[fit],total[fit])
                        for name,cs in cfgs.items():
                            for j,cfg in enumerate(cs):
                                if mode=='ann':p,s,_=deep.train('ANN',cfg,features,y[fit],seed+100*fold+ii)
                                else:p,s=old.predict(name,cfg,features,y[fit],seed+fold)
                                if name=='PLS_regression':regoof[j,innerlocal[ii][1]]=s
                                else:scores[name][j]+=old.base.balanced_accuracy_score(y[val],p)/4
                    for name,cs in cfgs.items():
                        for j,cfg in enumerate(cs):
                            conf=dict(region=region['name'],recipe=recipe,parameters=cfg)
                            if name=='PLS_regression':
                                score,threshold=choose_threshold(regoof[j],y[tr],innerlocal,midpoint);conf['threshold']=threshold
                            else:score=float(scores[name][j])
                            records.append(dict(model=name,inner_BA=score,config=json.dumps(conf,sort_keys=True)))
                            if score>best[name][0]+1e-12:best[name]=(score,conf)
                cache={}
                for name,(score,cfg) in best.items():
                    recipe=cfg['recipe']
                    if recipe not in cache:
                        blocks=prepare(x,w,tr,te,region,recipe)
                        cache[recipe]=deep.deep_inputs(blocks) if mode=='ann' else representations(blocks,y[tr],total[tr])
                    if mode=='ann':
                        p,s,diag=deep.train('ANN',cfg['parameters'],cache[recipe],y[tr],seed+100*fold+99)
                        cfg=dict(cfg,training=diag)
                    else:p,s=old.predict(name,cfg['parameters'],cache[recipe],y[tr],seed+fold)
                    if p is None:p=(s>=cfg['threshold']).astype(int)
                    assert np.isfinite(s).all()
                    before=len(preds)
                    row=append_result(rows,preds,name,rep,fold,te,y,ids,p,s,score,cfg)
                    row.update(region=region['name'],group=region['group'])
                    for pred in preds[before:]:pred.update(region=region['name'],group=region['group'])
                save(folder,rows,preds,records,rep,fold,region['name'])
            print(f'{mode} repeat={rep+1} fold={fold+1} 17 regions complete, elapsed={time.time()-start:.0f}s',flush=True)
        counts={}
        for v in caught:
            key=f'{v.category.__name__}: {v.message}';counts[key]=counts.get(key,0)+1
        (folder/'warnings.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
        assert not any(issubclass(v.category,old.ConvergenceWarning) for v in caught)
    print('COMPLETE '+mode,flush=True)

if __name__=='__main__':main(sys.argv[1])
