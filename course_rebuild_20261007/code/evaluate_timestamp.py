"""Compare course CF, metadata stacking, and timestamp features on fixed holdouts.

Random stacking uses instructor out-of-fold CF predictions. Forward stacking
uses earlier-time fits, never in-sample CF fitted values. Hyperparameters are
chosen using internal training validation or the global validation block.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json,joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from prepare import ROOT,load_ratings,save

ALPHAS=[1.,100.,10000.]
def metric(y,p,mean):
    e=np.asarray(p)-np.asarray(y)
    return {'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'OSR2':float(1-np.sum(e*e)/np.sum((np.asarray(y)-mean)**2))}

def history_features(fit,query):
    """Counts/gaps from fit-period timestamps strictly BEFORE each query time."""
    # Excluding equal timestamps avoids inventing an order within rating batches.
    histories={int(u):np.sort(g.timestamp.to_numpy()) for u,g in fit.groupby('user')}
    out=np.zeros((len(query),3),float)
    uq=query.user.to_numpy();tq=query.timestamp.to_numpy()
    for u,positions in query.groupby('user').indices.items():
        times=histories.get(int(u))
        if times is None:continue
        positions=np.asarray(positions);n=np.searchsorted(times,tq[positions],side='left')
        out[positions,0]=np.log1p(n)
        has=n>0;out[positions,1]=has
        gaps=(tq[positions[has]]-times[n[has]-1])/86400
        out[positions[has],2]=np.log1p(np.minimum(gaps,3650))
    return out

MOVIES=pd.read_csv(ROOT/'sources/movies.dat',sep='::',engine='python',encoding='latin-1',names=['movie','title','genres']).set_index('movie')
MOVIES['release_year']=MOVIES.title.str.extract(r'\((\d{4})\)\s*$',expand=False).astype(float)
USERS=pd.read_csv(ROOT/'sources/users.dat',sep='::',engine='python',names=['user','gender','age','occupation','zip']).set_index('user')

def make_features(rows,cf,fit):
    rows=rows.reset_index(drop=True);f=pd.DataFrame({'cf':np.asarray(cf),'release_year':rows.movie.map(MOVIES.release_year).to_numpy()})
    for col in ['gender','age','occupation']:f[col]=rows.user.map(USERS[col]).astype(str).to_numpy()
    f['genres']=rows.movie.map(MOVIES.genres).to_numpy()
    dt=pd.to_datetime(rows.timestamp,unit='s',utc=True)
    f['hour']=dt.dt.hour.astype(str);f['weekday']=dt.dt.weekday.astype(str)
    f['month_sin']=np.sin(2*np.pi*(dt.dt.month-1)/12)
    f['month_cos']=np.cos(2*np.pi*(dt.dt.month-1)/12)
    f['elapsed_days']=(rows.timestamp-float(fit.timestamp.min()))/86400
    h=history_features(fit,rows)
    for i,col in enumerate(['prior_count_log','has_prior_history','gap_days_log']):f[col]=h[:,i]
    return f

class FeatureEncoder:
    def __init__(self,timestamp):
        self.timestamp=timestamp
        self.numeric=['cf','release_year']+(['month_sin','month_cos','elapsed_days','prior_count_log','has_prior_history','gap_days_log'] if timestamp else [])
        self.categorical=['gender','age','occupation']+(['hour','weekday'] if timestamp else [])
    def fit(self,f):
        self.caps={k:(float(f[k].min()),float(f[k].max())) for k in ['elapsed_days','prior_count_log','gap_days_log']}
        self.scale=StandardScaler().fit(self.numbers(f))
        self.onehot=OneHotEncoder(handle_unknown='ignore',sparse_output=True).fit(f[self.categorical])
        self.genres=sorted({g for value in f.genres for g in value.split('|')})
        return self
    def numbers(self,f):
        a=f[self.numeric].to_numpy(float).copy()
        for j,k in enumerate(self.numeric):
            if k in getattr(self,'caps',{}):a[:,j]=np.clip(a[:,j],*self.caps[k])
        return a
    def transform(self,f):
        genre=np.column_stack([f.genres.str.split('|').map(lambda values:g in values).to_numpy(np.uint8) for g in self.genres])
        return sparse.hstack([sparse.csr_matrix(self.scale.transform(self.numbers(f))),self.onehot.transform(f[self.categorical]),sparse.csr_matrix(genre)],format='csr')
    def names(self):return self.numeric+list(self.onehot.get_feature_names_out(self.categorical))+['genre_'+g for g in self.genres]

def tune_and_fit(name,train_f,y,validation_f,vy,mean):
    results=[];choices={};models={}
    for timed,model_name in [(False,'CF + metadata'),(True,'CF + metadata + timestamp')]:
        encoder=FeatureEncoder(timed).fit(train_f)
        x=encoder.transform(train_f);v=encoder.transform(validation_f)
        for a in ALPHAS:
            model=Ridge(alpha=a,solver='lsqr',tol=1e-6).fit(x,y)
            p=np.clip(model.predict(v),1,5)
            results.append({'protocol':name,'model':model_name,'alpha':a,**metric(vy,p,mean)})
        winner=min([r for r in results if r['model']==model_name],key=lambda r:r['RMSE'])
        choices[model_name]=winner['alpha']
    pd.DataFrame(results).to_csv(ROOT/'results'/f'{name}_hybrid_validation.csv',index=False)
    return choices

def fit_final(name,train_f,y,test_f,choices):
    pred={}
    for timed,model_name in [(False,'CF + metadata'),(True,'CF + metadata + timestamp')]:
        encoder=FeatureEncoder(timed).fit(train_f);x=encoder.transform(train_f)
        model=Ridge(alpha=choices[model_name],solver='lsqr',tol=1e-6).fit(x,y)
        pred[model_name]=np.clip(model.predict(encoder.transform(test_f)),1,5)
        pd.DataFrame({'feature':encoder.names(),'coefficient':model.coef_}).to_csv(ROOT/'results'/f'{name}_{"timestamp" if timed else "metadata"}_coefficients.csv',index=False)
        # These are self-authored local model objects, not untrusted downloads.
        # Save state rather than a __main__ class so another process can load it.
        joblib.dump({'encoder_state':vars(encoder),'regression':model},ROOT/'models'/f'{name}_{"timestamp" if timed else "metadata"}.joblib')
    return pred

def load_hybrid(path):
    """Load a self-authored local artifact; never load an untrusted joblib file."""
    saved=joblib.load(path)
    state=saved['encoder_state'];encoder=FeatureEncoder(state['timestamp'])
    vars(encoder).update(state)
    return encoder,saved['regression']

def paired_ci(test,p0,p1,resamples=2000):
    # User-level resampling retains the dependence between a user's ratings.
    d=pd.DataFrame({'user':test.user.to_numpy(),'s0':(test.rating.to_numpy()-p0)**2,'s1':(test.rating.to_numpy()-p1)**2})
    g=d.groupby('user').agg(n=('s0','size'),s0=('s0','sum'),s1=('s1','sum'))
    arr=g[['n','s0','s1']].to_numpy();rng=np.random.default_rng(144);diff=[]
    for start in range(0,resamples,100):
        ix=rng.integers(0,len(arr),size=(min(100,resamples-start),len(arr)))
        total=arr[ix].sum(axis=1)
        diff.extend(np.sqrt(total[:,1]/total[:,0])-np.sqrt(total[:,2]/total[:,0]))
    return {'improvement_RMSE':float(np.sqrt(d.s0.mean())-np.sqrt(d.s1.mean())),'ci95':np.quantile(diff,[.025,.975]).tolist(),'resamples':resamples,'cluster':'user; event-weighted RMSE','users':len(g)}

def report_results(protocol,fit,test,pred,cf_warm):
    baseline=np.repeat(fit.rating.mean(),len(test));pred={'Training mean':baseline,**pred}
    metrics=[];detail=[]
    for name,p in pred.items():
        metrics.append({'protocol':protocol,'model':name,'scope':'all','rows':len(test),'users':test.user.nunique(),**metric(test.rating,p,fit.rating.mean())})
        for label,mask in [('warm',cf_warm),('cold',~cf_warm)]:
            if mask.any():metrics.append({'protocol':protocol,'model':name,'scope':label,'rows':int(mask.sum()),'users':test[mask].user.nunique(),**metric(test.rating.to_numpy()[mask],p[mask],fit.rating.mean())})
        detail.append(pd.DataFrame({'row':test.row.to_numpy(),'user':test.user.to_numpy(),'movie':test.movie.to_numpy(),'timestamp':test.timestamp.to_numpy(),'rating':test.rating.to_numpy(),'model':name,'prediction':p,'cf_warm':cf_warm}))
    pd.concat(detail).to_csv(ROOT/'results'/f'{protocol}_test_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    intervals={}
    for ref in ['softImpute','CF + metadata']:
        intervals[ref]=paired_ci(test,pred[ref],pred['CF + metadata + timestamp'])
    return metrics,intervals

def run():
    ratings=load_ratings();ratings['row']=np.arange(len(ratings))
    split=np.load(ROOT/'data/teacher_split.npz');tr=ratings.iloc[split['train']];te=ratings.iloc[split['test']]
    cf=np.load(ROOT/'data/teacher_oof_rank8.npy');f=make_features(tr,cf,tr)
    select,validation=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=144).split(f,groups=tr.user))
    choices=tune_and_fit('random',f.iloc[select],tr.rating.to_numpy()[select],f.iloc[validation],tr.rating.to_numpy()[validation],tr.rating.mean())
    testcf=pd.read_csv(ROOT/'results/teacher_rank8_predictions.csv').set_index('row').loc[te.row]
    tf=make_features(te,testcf.prediction.to_numpy(),tr)
    predictions={'softImpute':testcf.prediction.to_numpy(),**fit_final('random',f,tr.rating.to_numpy(),tf,choices)}
    metrics,intervals=report_results('random_99_1',tr,te,predictions,testcf.warm.to_numpy(bool))
    del f,tf

    chrono=np.load(ROOT/'data/temporal_split.npz');train=ratings.iloc[chrono['train']];val=ratings.iloc[chrono['validation']];test=ratings.iloc[chrono['test']]
    best=pd.read_csv(ROOT/'results/temporal_validation_grid.csv').sort_values('RMSE').iloc[0]
    validationcf=pd.read_csv(ROOT/'results'/f'temporal_r{int(best["rank"])}_l{int(best["lambda"])}_predictions.csv').set_index('row').loc[val.row]
    forward_features=[];forward_y=[]
    for key in ['forward_fold50','forward_fold65']:
        fold=pd.read_csv(ROOT/'results'/f'{key}_predictions.csv');query=ratings.iloc[fold.row.to_numpy()]
        history=ratings[ratings.timestamp<query.timestamp.min()]
        assert history.timestamp.max()<query.timestamp.min()
        forward_features.append(make_features(query,fold.prediction.to_numpy(),history));forward_y.append(query.rating.to_numpy())
    train_f=pd.concat(forward_features,ignore_index=True);train_y=np.concatenate(forward_y)
    val_f=make_features(val,validationcf.prediction.to_numpy(),train)
    choices2=tune_and_fit('temporal',train_f,train_y,val_f,val.rating.to_numpy(),train.rating.mean())
    # Final stack refit uses forward OOF predictions for all calibration events
    # in the 50%-90% period. The final holdout is never part of this refit.
    final_f=pd.concat([train_f,val_f],ignore_index=True);final_y=np.r_[train_y,val.rating.to_numpy()]
    fit=pd.concat([train,val]);testcf=pd.read_csv(ROOT/'results/temporal_final_predictions.csv').set_index('row').loc[test.row]
    test_f=make_features(test,testcf.prediction.to_numpy(),fit)
    predictions2={'softImpute':testcf.prediction.to_numpy(),**fit_final('temporal',final_f,final_y,test_f,choices2)}
    mm,ii=report_results('global_time_80_10_10',fit,test,predictions2,testcf.warm.to_numpy(bool));metrics+=mm
    table=pd.DataFrame(metrics);table.to_csv(ROOT/'results/model_metrics.csv',index=False)
    save('timestamp_experiment.json',{'predefined_alphas':ALPHAS,'teacher_rank':8,'temporal_selected_rank':int(best['rank']),'temporal_selected_lambda':int(best['lambda']),'random_hybrid_alphas':choices,'temporal_hybrid_alphas':choices2,'calendar_timezone':'UTC','temporal_stack_train_rows':len(train_f),'temporal_stack_final_rows':len(final_f),'random_stack':'Instructor out-of-fold rank-8 CF predictions; group validation for Ridge penalty','temporal_stack':'Two expanding-time CF fits and later validation predictions; no in-sample CF feature','history_features':'Fit-row timestamps strictly before query timestamp; tied times excluded','time_feature_caps':'Numeric drift/count/gap clipped to meta-fit observed ranges','intervals':{'random':intervals,'temporal':ii},'all_metrics':metrics})
    print(table[table.scope=='all'].to_string(index=False),flush=True)
    print(json.dumps({'random':intervals,'temporal':ii},indent=2),flush=True)

if __name__=='__main__':run()
