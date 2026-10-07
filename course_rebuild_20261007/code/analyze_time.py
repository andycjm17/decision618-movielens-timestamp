"""Timestamp EDA and prediction drift tables. All dates below are UTC."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from prepare import ROOT,load_ratings,save

def analyze():
    r=load_ratings();r['row']=np.arange(len(r));r['date']=pd.to_datetime(r.timestamp,unit='s',utc=True)
    r['month']=r.date.dt.strftime('%Y-%m')
    monthly=r.groupby('month').agg(ratings=('rating','size'),users=('user','nunique'),mean_rating=('rating','mean')).reset_index()
    monthly.to_csv(ROOT/'results/monthly_activity.csv',index=False)
    ordered=r.sort_values(['user','timestamp','movie']);gap=ordered.groupby('user').timestamp.diff().dropna()
    stats={'adjacent_pairs':len(gap),'equal_second_pairs':int((gap==0).sum()),'equal_second_fraction':float((gap==0).mean()),'within_60_seconds_fraction':float((gap<=60).mean()),'within_30_minutes_fraction':float((gap<=1800).mean()),'median_gap_seconds':float(gap.median()),'timestamp_meaning':'Time a rating was entered, not time a movie was viewed','calendar_timezone':'UTC; user local time not available'}
    save('timestamp_eda.json',stats)
    hist=pd.DataFrame({'label':['same second','1-60 seconds','61 seconds-30 minutes','30 minutes-1 day','over 1 day'],
                       'pairs':[int((gap==0).sum()),int(((gap>0)&(gap<=60)).sum()),int(((gap>60)&(gap<=1800)).sum()),int(((gap>1800)&(gap<=86400)).sum()),int((gap>86400).sum())]})
    hist['fraction']=hist.pairs/len(gap);hist.to_csv(ROOT/'results/adjacent_time_gaps.csv',index=False)
    # A 30-minute gap defines a rating-entry batch, not an inferred watch session.
    start=ordered.groupby('user').timestamp.diff().isna() | (ordered.groupby('user').timestamp.diff()>1800)
    ordered['batch']=start.cumsum();batches=ordered.groupby('batch').agg(user=('user','first'),events=('rating','size'),first=('timestamp','min'),last=('timestamp','max'))
    save('rating_batches.json',{'gap_threshold_seconds':1800,'batches':len(batches),'median_ratings':float(batches.events.median()),'p90_ratings':float(batches.events.quantile(.9)),'definition':'Descriptive rating-entry batches; no claim about viewing sessions'})
    # Period and known-ID drift. The final fitted model uses the first 90%.
    split=np.load(ROOT/'data/temporal_split.npz');first=r.iloc[split['train']];val=r.iloc[split['validation']];last=r.iloc[split['test']];fit=pd.concat([first,val])
    overview=[]
    for name,q,history in [('train',first,first),('validation',val,first),('test',last,fit)]:
        warm=q.user.isin(history.user)&q.movie.isin(history.movie)
        overview.append({'split':name,'ratings':len(q),'users':q.user.nunique(),'mean_rating':q.rating.mean(),'rating_std':q.rating.std(),'warm_rows':int(warm.sum()),'cold_rows':int((~warm).sum()),'warm_fraction':warm.mean()})
    pd.DataFrame(overview).to_csv(ROOT/'results/temporal_population.csv',index=False)
    p=pd.read_csv(ROOT/'results/global_time_80_10_10_test_predictions.csv.gz')
    p['period']=pd.to_datetime(p.timestamp,unit='s',utc=True).dt.strftime('%Y-%m')
    period=[]
    for (key,model),q in p.groupby(['period','model']):
        e=q.prediction-q.rating
        period.append({'period':key,'model':model,'rows':len(q),'users':q.user.nunique(),'mean_rating':q.rating.mean(),'warm_fraction':q.cf_warm.mean(),'RMSE':np.sqrt(np.mean(e**2)),'MAE':abs(e).mean(),'mean_prediction':q.prediction.mean(),'clipped_boundary_fraction':q.prediction.isin([1.,5.]).mean()})
    pd.DataFrame(period).to_csv(ROOT/'results/temporal_test_by_month.csv',index=False)
    # Match instructor cells 17-23: genre means, row max-abs scaling and signs.
    mapping=pd.read_csv(ROOT/'data/teacher_movie_mapping.csv')
    movies=pd.read_csv(ROOT/'sources/movies.dat',sep='::',engine='python',encoding='latin-1',names=['movie','title','genres'])
    v=pd.read_csv(ROOT/'results/teacher_rank8_movie_factors.csv').merge(mapping,on='movieID',validate='one_to_one').merge(movies,on='movie',validate='one_to_one')
    genres=sorted({g for value in v.genres for g in value.split('|')});factor_cols=[f'X{k}' for k in range(1,9)]
    table=np.column_stack([v.loc[v.genres.str.split('|').map(lambda x:g in x),factor_cols].mean().to_numpy() for g in genres])
    scale=np.max(abs(table),axis=1,keepdims=True);table=np.divide(table,scale,out=np.zeros_like(table),where=scale!=0)*5
    table[table.sum(axis=1)>0]*=-1
    pd.DataFrame(table,index=[f'Factor {i}' for i in range(1,9)],columns=genres).to_csv(ROOT/'results/genre_factor_heatmap.csv')
    # Titles in Canvas were accent-normalized; all IDs and genres are unchanged.
    reference=ROOT.parent/'final_project/data/raw/movies.dat'
    if reference.exists():
        original=pd.read_csv(reference,sep='::',engine='python',encoding='latin-1',names=['movie','title','genres'])
        compare=movies.merge(original,on='movie',suffixes=('_canvas','_grouplens'),validate='one_to_one')
        changed=compare.title_canvas!=compare.title_grouplens
        assert (compare.genres_canvas==compare.genres_grouplens).all()
        compare[changed].to_csv(ROOT/'results/movie_title_normalization.csv',index=False)
        years=lambda s:s.str.extract(r'\((\d{4})\)\s*$',expand=False)
        assert years(compare.title_canvas).equals(years(compare.title_grouplens))
        save('movie_source_comparison.json',{'same_IDs':len(compare)==len(movies)==len(original),'same_genres':True,'same_release_years':True,'changed_titles':int(changed.sum()),'title_differences':'Mostly ASCII normalization; also character and punctuation simplifications','model_features_unchanged':True})
    print(json.dumps(stats,indent=2))
    print(pd.DataFrame(overview).to_string(index=False))

if __name__=='__main__':analyze()
