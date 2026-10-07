"""Independent deterministic integrity checks on splits, features and metrics."""
import json,hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error
from prepare import ROOT,load_ratings,save
from evaluate_timestamp import make_features,history_features,load_hybrid

def audit(require_fresh=True,verbose=True):
    r=load_ratings();r['row']=np.arange(len(r));checks={}
    random=np.load(ROOT/'data/teacher_split.npz');t=np.load(ROOT/'data/temporal_split.npz')
    checks['random_disjoint_exhaustive']=(len(np.intersect1d(random['train'],random['test']))==0 and len(np.union1d(random['train'],random['test']))==len(r))
    checks['temporal_disjoint_exhaustive']=len(np.unique(np.concatenate([t[x] for x in t.files])))==len(r) and sum(len(t[x]) for x in t.files)==len(r)
    checks['temporal_strict_boundaries']=r.iloc[t['train']].timestamp.max()<r.iloc[t['validation']].timestamp.min() and r.iloc[t['validation']].timestamp.max()<r.iloc[t['test']].timestamp.min()
    checks['exact_legacy_random_rows']=random['test'][:5].tolist()==[99,238,283,363,417]
    checks['all_20_instructor_cache_scores_match']=json.loads((ROOT/'results/cache_audit.json').read_text())['all_20_metric_rows_reproduced']
    metrics=pd.read_csv(ROOT/'results/model_metrics.csv');metric_error=[]
    for protocol,fit,query in [('random_99_1',r.iloc[random['train']],r.iloc[random['test']]),('global_time_80_10_10',r.iloc[np.r_[t['train'],t['validation']]],r.iloc[t['test']])]:
        pred=pd.read_csv(ROOT/'results'/f'{protocol}_test_predictions.csv.gz')
        for name,q in pred.groupby('model'):
            assert q.row.to_numpy().tolist()==query.row.to_numpy().tolist()
            assert np.array_equal(q.rating.to_numpy(),query.rating.to_numpy())
            assert np.isfinite(q.prediction).all() and q.prediction.between(1,5).all()
            for scope,mask in [('all',np.ones(len(q),bool)),('warm',q.cf_warm.to_numpy(bool)),('cold',~q.cf_warm.to_numpy(bool))]:
                if not mask.any():continue
                z=q[mask];y=z.rating.to_numpy();p=z.prediction.to_numpy()
                result=[np.sqrt(mean_squared_error(y,p)),mean_absolute_error(y,p),1-np.sum((y-p)**2)/np.sum((y-fit.rating.mean())**2)]
                saved=metrics[(metrics.protocol==protocol)&(metrics.model==name)&(metrics.scope==scope)].iloc[0]
                metric_error.append(max(abs(np.asarray(result)-saved[['RMSE','MAE','OSR2']].to_numpy(float))))
        # Round-trip stored encoders/models; never fit a transform to test data.
        cf=pred[pred.model=='softImpute'].prediction.to_numpy()
        ix=np.arange(0,len(query),max(1,len(query)//500));sample=query.iloc[ix]
        f=make_features(sample,cf[ix],fit)
        # Prediction features do not read query rating labels.
        changed=sample.copy();changed['rating']=6-changed.rating
        checks[protocol+'_features_ignore_query_labels']=f.equals(make_features(changed,cf[ix],fit))
        key='random' if protocol=='random_99_1' else 'temporal'
        for timed,model in [(False,'CF + metadata'),(True,'CF + metadata + timestamp')]:
            encoder,reg=load_hybrid(ROOT/'models'/f'{key}_{"timestamp" if timed else "metadata"}.joblib')
            p=np.clip(reg.predict(encoder.transform(f)),1,5)
            actual=pred[pred.model==model].prediction.to_numpy()[ix]
            checks[key+'_'+str(timed)+'_portable_prediction']=bool(np.max(abs(p-actual))<1e-10)
        h=history_features(fit,sample)
        for j,(_,row) in enumerate(sample.iterrows()):
            past=fit.loc[(fit.user==row.user)&(fit.timestamp<row.timestamp),'timestamp']
            expected=[np.log1p(len(past)),float(len(past)>0),np.log1p(min((row.timestamp-past.max())/86400,3650)) if len(past) else 0]
            assert np.allclose(h[j],expected,atol=1e-12)
    checks['independent_metrics_all_24_rows']=max(metric_error)<1e-10
    # Explicit same-second and unknown-user feature examples.
    fit=pd.DataFrame({'user':[1,1,1],'timestamp':[10,20,20]})
    query=pd.DataFrame({'user':[1,1,1,2],'timestamp':[10,20,21,30]})
    h=history_features(fit,query)
    checks['same_second_events_excluded']=np.allclose(h[:,0],np.log1p([0,1,3,0])) and np.array_equal(h[:,1],[0,1,1,0])
    checks['matrix_factor_predictions_independently_verified']=(pd.read_csv(ROOT/'results/independent_r_audit.csv').filter(like='error').to_numpy()<1e-10).all()
    for name in ['forward_fold50','forward_fold65']:
        q=pd.read_csv(ROOT/'results'/f'{name}_predictions.csv');checks[name+'_rows_in_meta_training_window']=set(q.row).issubset(set(t['train']))
        checks[name+'_no_future_test_rows']=not set(q.row).intersection(t['test'])
    if require_fresh:
        log=pd.read_csv(ROOT/'results/fresh_cv_fit_log.csv');fresh=pd.read_csv(ROOT/'results/fresh_cv_metrics.csv')
        checks['fresh_400_unique_fits_complete']=len(log)==400 and len(log[['fold','rank']].drop_duplicates())==400
        checks['fresh_fits_no_recorded_warnings']=log.warnings.fillna('').eq('').all()
        checks['fresh_cv_20_finite_scores']=len(fresh)==20 and np.isfinite(fresh[['RMSE','MAE','OSR2']]).all().all()
    checks={k:bool(v) for k,v in checks.items()};assert all(checks.values()),checks
    audit_result={'passed':True,'checks':checks,'max_independent_metric_error':float(max(metric_error)),'prediction_labels_checked':'All model predictions aligned with raw row IDs and original labels','causal_claim':False,'history_scope':'Frozen fit-period history only; no within-holdout online updates'}
    save('analysis_audit.json',audit_result)
    if verbose:print(json.dumps(audit_result,indent=2))
    return audit_result

if __name__=='__main__':audit()
