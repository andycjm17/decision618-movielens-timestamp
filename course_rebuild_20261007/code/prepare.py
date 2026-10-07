"""Prepare exact instructor random split, audit cached CV, and forward-time split."""
from pathlib import Path
import pickle,importlib,json,hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def save(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

class DataOnlyUnpickler(pickle.Unpickler):
    # The teaching cache is untrusted binary data. Permit only the constructors
    # needed by its NumPy arrays and Pandas frame, never arbitrary functions.
    ALLOWED={('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),
        ('pandas._libs.internals','_unpickle_block'),('numpy._core.multiarray','_reconstruct'),
        ('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),
        ('builtins','slice'),('pandas.core.indexes.base','_new_Index'),
        ('pandas.core.indexes.base','Index'),('pandas.core.indexes.range','RangeIndex')}
    def find_class(self,module,name):
        if (module,name) not in self.ALLOWED:raise pickle.UnpicklingError(f'Unsupported cache constructor {module}.{name}')
        if module=='numpy._core.multiarray':module='numpy.core.multiarray'
        return getattr(importlib.import_module(module),name)

def load_ratings():
    return pd.read_csv(ROOT/'sources/ratings.dat',sep='::',engine='python',names=['user','movie','rating','timestamp'])

def prepare():
    for d in ['data','results','models','figures']: (ROOT/d).mkdir(exist_ok=True)
    r=load_ratings();r['row']=np.arange(len(r),dtype=np.int32)
    assert len(r)==1000209 and not r.duplicated(['user','movie']).any()
    assert r.rating.between(1,5).all() and not r.isna().any().any()
    movies=pd.read_csv(ROOT/'sources/movies.dat',sep='::',engine='python',encoding='latin-1',names=['movie','title','genres'])
    users=pd.read_csv(ROOT/'sources/users.dat',sep='::',engine='python',names=['user','gender','age','occupation','zip'])
    assert movies.movie.is_unique and users.user.is_unique
    assert r.movie.isin(movies.movie).all() and r.user.isin(users.user).all()
    movie_ids=sorted(r.movie.unique());mapping={m:i+1 for i,m in enumerate(movie_ids)}
    # Legacy RandomState reproduces np.random.seed + np.random.choice, not
    # default_rng, which intentionally gives a different random split.
    rng=np.random.RandomState(144)
    tr=np.sort(rng.choice(np.arange(len(r)),size=int(.99*len(r)),replace=False))
    te=np.setdiff1d(np.arange(len(r)),tr)
    folds=np.random.RandomState(144).randint(1,21,size=len(tr))
    assert len(tr)==990206 and len(te)==10003 and te[:5].tolist()==[99,238,283,363,417]
    np.savez_compressed(ROOT/'data/teacher_split.npz',train=tr,test=te,cv_fold=folds)
    r.to_csv(ROOT/'data/ratings.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    rr=r.copy();rr['userID']=rr.user;rr['movieID']=rr.movie.map(mapping)
    rr[['row','userID','movieID','rating','timestamp']].to_csv(ROOT/'data/ratings_for_r.csv',index=False)
    pd.DataFrame({'row':tr,'fold':folds}).to_csv(ROOT/'data/teacher_train.csv',index=False)
    pd.DataFrame({'row':te}).to_csv(ROOT/'data/teacher_test.csv',index=False)
    pd.DataFrame({'movie':movie_ids,'movieID':range(1,len(movie_ids)+1)}).to_csv(ROOT/'data/teacher_movie_mapping.csv',index=False)
    with (ROOT/'sources/cv_all_1m.pkl').open('rb') as f: cv=DataOnlyUnpickler(f).load()
    info=cv['info'];assert len(info)==20 and len(cv['pred'])==20
    y=r.iloc[tr].rating.to_numpy(float)
    fold_mean={k:y[folds!=k].mean() for k in range(1,21)}
    baseline=np.array([fold_mean[k] for k in folds]);denom=np.sum((y-baseline)**2)
    checks=[]; recomputed=[]
    for j,p in enumerate(cv['pred']):
        p=np.asarray(p,float);assert p.shape==(len(tr),) and np.isfinite(p).all()
        assert p.min()>=1 and p.max()<=5
        e=p-y;sse=float(e@e)
        metrics={'archetypes':int(info.iloc[j].archetypes),'r2':1-sse/denom,'SSE':sse,'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e)))}
        checks.append(all(np.isclose(metrics[k],info.iloc[j][k],atol=1e-8,rtol=1e-10) for k in ['r2','SSE','RMSE','MAE']))
        recomputed.append(metrics)
    assert all(checks),'The supplied cache does not match the reconstructed split/folds.'
    table=pd.DataFrame(recomputed);rank=int(table.loc[table.r2.idxmax(),'archetypes'])
    assert rank==8
    table.to_csv(ROOT/'results/teacher_cv_recomputed.csv',index=False)
    np.save(ROOT/'data/teacher_oof_rank8.npy',np.asarray(cv['pred'][rank-1],dtype=float))
    save('cache_audit.json',{'all_20_metric_rows_reproduced':True,'selected_rank':rank,'folds':20,'train_rows':len(tr),'test_rows':len(te),'seed':144,'first_test_rows':te[:5].tolist(),'cache_reused_not_400_models_retrained':True,'original_R_seed_not_set':True,'cache_sha256':hashlib.sha256((ROOT/'sources/cv_all_1m.pkl').read_bytes()).hexdigest()})
    # Timestamp-only quantiles choose event volume boundaries, before looking at
    # ratings. Entire tied timestamp groups stay on one side of each boundary.
    t=r.timestamp.to_numpy();ordered=np.sort(t)
    a=int(ordered[int(.80*len(t))]);b=int(ordered[int(.90*len(t))])
    idx={'train':np.flatnonzero(t<a),'validation':np.flatnonzero((t>=a)&(t<b)),'test':np.flatnonzero(t>=b)}
    for k,v in idx.items():pd.DataFrame({'row':v}).to_csv(ROOT/'data'/f'temporal_{k}.csv',index=False)
    np.savez_compressed(ROOT/'data/temporal_split.npz',**idx)
    assert t[idx['train']].max()<t[idx['validation']].min()<t[idx['test']].min()
    assert t[idx['validation']].max()<t[idx['test']].min()
    splits=[]
    for k,ix in idx.items():
        part=r.iloc[ix];splits.append({'split':k,'rows':len(part),'first_utc':pd.to_datetime(part.timestamp.min(),unit='s',utc=True).isoformat(),'last_utc':pd.to_datetime(part.timestamp.max(),unit='s',utc=True).isoformat(),'users':part.user.nunique(),'movies':part.movie.nunique()})
    save('temporal_split.json',{'boundary_rule':'timestamp-only 80th/90th percentiles; left-closed later partitions; ties never split','train_cutoff':a,'test_cutoff':b,'partitions':splits,'target':'Explicit 1-5 rating, unchanged from the course'})
    # Different missing-rating completion and forecasting tasks need different
    # splits. Quantify chronology violations in the random classroom benchmark.
    latest_train=r.iloc[tr].groupby('user').timestamp.max()
    test=r.iloc[te];future=(test.user.map(latest_train)>test.timestamp).mean()
    save('preparation.json',{'ratings':len(r),'users':r.user.nunique(),'rated_movies':r.movie.nunique(),'catalog_movies':len(movies),'random_test_has_later_same_user_training_fraction':future,'raw_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'sources/ratings.dat',ROOT/'sources/movies.dat',ROOT/'sources/users.dat']}})
    print('Prepared exact random split and checked all 20 cached CV metric rows. Rank:',rank,flush=True)
    print(json.dumps(splits,indent=2),flush=True)

if __name__=='__main__':prepare()
