import warnings, numpy as np, pandas as pd; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from predict import COLS, cross_validate, priority
from featurize import build_features, CORE
from model import CLASS_TO_ORD
from pipeline import load, BASE, build_evidence
D=load(); cand=D['cand'].reset_index(drop=True); n=len(cand)
ev=build_evidence(D,verbose=False); t1=pd.to_datetime(D['up'].observed_date).max()
F0=build_features(ev,cand,"T0",t1); F1=build_features(ev,cand,"T1",t1)
lab=pd.read_csv(BASE+"Development_Labels/Development_Labels.csv")
pos={c:i for i,c in enumerate(cand.candidate_record_id)}; li=np.array([pos[c] for c in lab.candidate_record_id])
X=pd.concat([F0.iloc[li],F1.iloc[li]],ignore_index=True)[COLS]
y=np.array([CLASS_TO_ORD[v] for v in list(lab.label_t0)+list(lab.label_t1)])
grp=np.concatenate([li,li]); ph=np.array([0]*300+[1]*300)
o=cross_validate(X,y,grp)
print("PER PHASE")
for p,nm in [(0,'T0'),(1,'T1')]:
    m=ph==p
    print(f"  {nm}: ll={log_loss(y[m],o[m],labels=[0,1,2]):.4f} acc={accuracy_score(y[m],o[m].argmax(1)):.4f} "
          f"auc={np.mean([roc_auc_score((y[m]==k).astype(int),o[m][:,k]) for k in range(3)]):.4f} "
          f"prio={roc_auc_score((y[m]==2).astype(int),priority(o[m])):.4f}")
print("\nCONFUSION (rows true, cols pred; notw/insuf/warr)"); print(confusion_matrix(y,o.argmax(1)))
print("\nCALIBRATION quintiles")
for k,nm in [(2,'p_warranted'),(0,'p_not_warranted')]:
    b=pd.qcut(o[:,k],5,labels=False,duplicates='drop')
    t=pd.DataFrame({'p':o[:,k],'obs':(y==k).astype(int),'b':b}).groupby('b').agg(pred=('p','mean'),obs=('obs','mean'))
    print(f"  {nm}: "+"  ".join(f"{r.pred:.2f}/{r.obs:.2f}" for r in t.itertuples()))
print("\nPER-FEED rho (T0, 150d half-life)")
ref=pd.to_datetime(cand.candidate_observed_date).values
e=ev[(~ev.is_t1)&ev.state.notna()].copy()
e["age"]=((ref[e.cand.values]-e.date.values)/np.timedelta64(1,"D")).clip(min=0)
e["de"]=(e.state=="DE").astype(float); y0=lab.label_t0.map(CLASS_TO_ORD).values
def rho(v):
    v=np.asarray(v,float)[li]; m=~np.isnan(v); return spearmanr(v[m],y0[m]).statistic
def sc(sub):
    w=sub.w*0.5**(sub.age/150)
    num=np.bincount(sub.cand,weights=w*sub.de,minlength=n); den=np.bincount(sub.cand,weights=w,minlength=n)
    return np.where(den>1e-9,num/np.maximum(den,1e-9),np.nan)
for s in CORE+['wrk']: print(f"  {s}: {rho(sc(e[e.stream==s])):+.3f}")
print(f"  core pooled: {rho(sc(e[e.stream.isin(CORE)])):+.3f}")
op=e[(e.stream=='adr')&(e.open_ended==True)]
print(f"  current address only: {rho(sc(op)):+.3f}")
print(f"\nsubmission class mix: {pd.read_csv('case_predictions.csv').predicted_class.value_counts().to_dict()}")
