"""Small ANN, 1D-CNN and Transformer controls; no test-set early stopping."""
from all_algorithm_suite import *
sys.path.append(r'C:\Users\22258006\.codex\visualizations\2026\09\28\01a0e676-9363-70a2-9c5d-8e7f7d545738\ml_deps')
import torch
from torch import nn
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
CONFIG=[dict(width=16,lr=.001),dict(width=32,lr=.003)]
FAMILIES=['ANN','CNN_1D','Transformer']

def deep_inputs(blocks):
    pooled=[]
    for a,b in blocks:
        pos=np.linspace(0,1,a.shape[1]);grid=np.linspace(0,1,128)
        pooled.append(tuple(np.array([np.interp(grid,pos,row) for row in x],dtype=np.float32) for x in [a,b]))
    sequence=tuple(np.stack([v[j] for v in pooled],axis=1) for j in [0,1])
    a,b=block_concat(blocks);m=PCA(n_components=10,svd_solver='full').fit(a);a=m.transform(a);b=m.transform(b);sc=StandardScaler().fit(a)
    return dict(sequence=sequence,ann=(sc.transform(a).astype(np.float32),sc.transform(b).astype(np.float32)))

class CNN(nn.Module):
    def __init__(self,blocks,width):
        super().__init__();self.branches=nn.ModuleList([nn.Sequential(nn.Conv1d(1,8,5,padding=2),nn.ReLU(),nn.AvgPool1d(2),nn.Conv1d(8,16,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(8),nn.Flatten()) for _ in range(blocks)])
        self.head=nn.Sequential(nn.Linear(blocks*128,width),nn.ReLU(),nn.Dropout(.3),nn.Linear(width,1))
    def forward(self,x):return self.head(torch.cat([m(x[:,j:j+1]) for j,m in enumerate(self.branches)],dim=1)).ravel()

class Transformer(nn.Module):
    def __init__(self,blocks,width):
        super().__init__();self.embed=nn.Linear(8,width);self.pos=nn.Parameter(torch.zeros(1,blocks*16,width))
        layer=nn.TransformerEncoderLayer(d_model=width,nhead=2,dim_feedforward=2*width,dropout=.2,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False);self.head=nn.Linear(width,1)
    def forward(self,x):
        z=x.reshape(x.shape[0],-1,8);z=self.embed(z)+self.pos
        return self.head(self.encoder(z).mean(dim=1)).ravel()

def train(family,cfg,data,y,seed):
    torch.manual_seed(seed);np.random.seed(seed)
    a,b=data['ann' if family=='ANN' else 'sequence'];a=torch.from_numpy(a);b=torch.from_numpy(b);yy=torch.tensor(y,dtype=torch.float32)
    if family=='ANN':model=nn.Sequential(nn.Linear(10,cfg['width']),nn.ReLU(),nn.Dropout(.3),nn.Linear(cfg['width'],1))
    elif family=='CNN_1D':model=CNN(a.shape[1],cfg['width'])
    else:model=Transformer(a.shape[1],cfg['width'])
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y==0).sum()/(y==1).sum())))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01);first=None
    for epoch in range(60):
        model.train();opt.zero_grad();loss=lossfn(model(a).ravel(),yy);assert torch.isfinite(loss)
        if first is None:first=float(loss.detach())
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    model.eval()
    with torch.no_grad():s=torch.sigmoid(model(b).ravel()).numpy();ts=torch.sigmoid(model(a).ravel()).numpy()
    return (s>=.5).astype(int),s,dict(first_loss=first,last_loss=float(loss.detach()),train_BA=float(balanced_accuracy_score(y,ts>=.5)),parameters=sum(v.numel() for v in model.parameters()))

def main(cohort):
    folder=OUT/(cohort+'_deep');folder.mkdir(exist_ok=True);raw,w,y,ids=inputs(cohort);innern=4 if cohort=='nir' else 3
    protocol=dict(cohort=cohort,seeds=SEEDS,outer=5,inner=innern,families=FAMILIES,config=CONFIG,epochs=60,batch='full training fold',optimizer='AdamW weight_decay .01, weighted BCE, gradient clip1',preprocessing='Same two recipes as classical suite; row interpolation to128 points separately per modality for CNN/Transformer; ANN training PCA10 plus standardization',CNN='Separate modality branches Conv1D8/16, kernel5, adaptive pooling8, concatenated dense width16/32',Transformer='One encoder layer, 2 heads, patch8 points, width16/32, positional embedding, dropout.2, mean pool; fusion concatenates modality token sequences',ANN='PCA10 -> hidden16/32 ReLU dropout.3 -> binary output',selection='60 fixed epochs, no early stopping/test monitoring; recipe/width/lr chosen by inner BA; one deterministic initialization per fit; small network controls, no pretraining or augmentation',torch=torch.__version__)
    (folder/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8');rows=[];preds=[];splits=[];started=time.time()
    for repeat,seed in enumerate(SEEDS):
        for fold,(tr,te) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(raw[0],y)):
            inner=list(StratifiedKFold(innern,shuffle=True,random_state=seed+fold+1).split(raw[0][tr],y[tr]));best={f:(-np.inf,None) for f in FAMILIES};inner_records=[]
            for recipe in [0,1]:
                scores={f:np.zeros(len(CONFIG)) for f in FAMILIES}
                for ii,(it,iv) in enumerate(inner):
                    fit=tr[it];val=tr[iv];assert not set(fit)&set(np.r_[val,te]);data=deep_inputs(prep_blocks(raw,w,fit,val,recipe,cohort))
                    for family in FAMILIES:
                        for j,cfg in enumerate(CONFIG):
                            pp,_,_=train(family,cfg,data,y[fit],seed+100*fold+ii);scores[family][j]+=(np.mean(pp[y[val]==0]==0)+np.mean(pp[y[val]==1]==1))/(2*innern)
                for family in FAMILIES:
                    for j,cfg in enumerate(CONFIG):
                        score=float(scores[family][j]);conf=dict(recipe=recipe,parameters=cfg);inner_records.append(dict(model=family,inner_BA=score,config=json.dumps(conf,sort_keys=True)))
                        if score>best[family][0]+1e-12:best[family]=(score,conf)
            cache={}
            for family,(score,cfg) in best.items():
                r=cfg['recipe']
                if r not in cache:cache[r]=deep_inputs(prep_blocks(raw,w,tr,te,r,cohort))
                pp,ss,diagnostics=train(family,cfg['parameters'],cache[r],y[tr],seed+100*fold+99)
                append_result(rows,preds,family,repeat,fold,te,y,ids,pp,ss,score,dict(cfg,training=diagnostics))
            pd.DataFrame(rows).to_csv(folder/'fold_metrics.csv',index=False);pd.DataFrame(preds).to_csv(folder/'oof_predictions.csv',index=False);pd.DataFrame(inner_records).to_csv(folder/f'inner_r{repeat}_f{fold}.csv',index=False)
            splits.append(dict(repeat=repeat,fold=fold,train_ids=ids[tr].tolist(),test_ids=ids[te].tolist(),inner=[dict(train_ids=ids[tr[it]].tolist(),validation_ids=ids[tr[iv]].tolist()) for it,iv in inner]));(folder/'splits.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
            print(f'deep {cohort} {repeat+1}/{fold+1}; {time.time()-started:.0f}s; '+', '.join(f'{r["model"]}={r["BA"]:.3f}' for r in rows[-3:]),flush=True)
    print('deep '+cohort+' complete',flush=True)

if __name__=='__main__':
    for cohort in (['nir','raman','fusion'] if sys.argv[1]=='all' else [sys.argv[1]]):main(cohort)
