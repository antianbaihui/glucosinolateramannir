"""Isolated corrected head-only rerun; historic scripts and outputs preserved."""
import sys,json
from pathlib import Path
MODE=sys.argv[1]; COHORT=sys.argv[2] if len(sys.argv)>2 else 'nir'
import classify_74samples as c
np,pd=c.np,c.pd
ROOT=Path(__file__).resolve().parent/'head_confirmed_20261007'
c.OUT=ROOT/'all_algorithms';c.OUT.mkdir(exist_ok=True)
labels=pd.read_csv(ROOT/'head_74_labels.csv',dtype={'ID':str})
heads=pd.read_csv(ROOT/'head_spectra_126.csv',dtype={'ID':str}).set_index('ID')
w=np.array(heads.columns[1:],float)
x=heads.loc[labels.ID].iloc[:,1:].to_numpy(float)
y=labels.label.to_numpy();ids=labels.ID.to_numpy();total=labels.Total_mg_per_g_DW.to_numpy()
assert x.shape==(74,1557) and np.isfinite(x).all()
def submitted():
    return labels.rename(columns={'Total_mg_per_g_DW':'Total'})[['ID','Total','label']]
c.submitted=submitted
raman=None
def inputs(cohort):
    global raman
    mask=(w>=4000)&(w<=7500)
    if cohort=='nir':return [x[:,mask]],[w[mask]],y,ids
    if raman is None:
        raman=c.base.inputs('raman') # Raman branch returns only Raman; no merged NIR used.
    rr,rw,ry,ri=raman;ri=ri.astype(str)
    ix=np.array([i for i,sid in enumerate(ids) if sid in ri])
    jx=np.array([np.flatnonzero(ri==sid)[0] for sid in ids[ix]])
    assert len(ix)==37 and np.array_equal(y[ix],ry[jx])
    if cohort=='raman':return [rr[0][jx]],rw,y[ix],ids[ix]
    if cohort=='nir_paired':return [x[ix][:,mask]],[w[mask]],y[ix],ids[ix]
    return [x[ix][:,mask],rr[0][jx]],[w[mask],rw[0]],y[ix],ids[ix]
c.inputs=inputs
if MODE.startswith('segment'):
    import segment_74samples as s
    s.OUT=ROOT/'segmentation';s.OUT.mkdir(exist_ok=True)
    s.inputs=lambda:(x,w,y,total,ids)
    if MODE=='segment_report':
        import report_segmentation_74samples as report
        report.OUT=s.OUT;report.main()
    else:
        c.OUT=ROOT/'segment_splits';(c.OUT/'nir').mkdir(parents=True,exist_ok=True)
        (c.OUT/'nir/splits.json').write_bytes((ROOT/'splits.json').read_bytes())
        s.main(MODE.split('_')[1])
elif MODE=='report':
    import hashlib
    sources=[ROOT/'head_spectra_126.csv',ROOT/'head_74_labels.csv',ROOT/'audit.json']
    audit={'input_sha256':{str(p.relative_to(ROOT.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (c.OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    for cohort in c.COHORTS:
        _,_,_,ii=inputs(cohort)
        submitted().set_index('ID').loc[ii].reset_index().to_csv(c.OUT/f'{cohort}_samples.csv',index=False)
    import report_74samples as report
    report.OUT=c.OUT;report.main()
else:
    for cohort in (c.COHORTS if COHORT=='all' else [COHORT]):
        getattr(c,MODE)(cohort)
        folder=c.OUT/(cohort+('_'+MODE if MODE in ['deep','tabpfn'] else ''))
        p=folder/'protocol.json'
        if p.exists():
            info=json.loads(p.read_text(encoding='utf8'))
            info.update(corrected_head_only=True,head_suffix='-h.spa',unit='mg/g DW',source_audit=str(ROOT/'audit.json'))
            p.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
