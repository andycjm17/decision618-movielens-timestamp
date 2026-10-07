"""Editable Matplotlib figures for the report and executed notebook."""
import os
from prepare import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime/mpl-cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'runtime/font-cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLORS=['#8894a4','#be4939','#ed9753','#277b83']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':175,'figure.facecolor':'white','axes.facecolor':'white','axes.axisbelow':True})
MODELS=['Training mean','softImpute','CF + metadata','CF + metadata + timestamp']

def finish(fig,name):
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight')
    return fig

def model_chart():
    d=pd.read_csv(ROOT/'results/model_metrics.csv');d=d[d.scope=='all']
    fig,axs=plt.subplots(1,2,figsize=(11.8,4.2),sharey=True)
    for ax,protocol,title in zip(axs,['random_99_1','global_time_80_10_10'],['Classroom random holdout | n = 10,003','Future holdout | n = 100,021']):
        values=d[d.protocol==protocol].set_index('model').loc[MODELS].RMSE.to_numpy()
        ax.bar(np.arange(4),values,color=COLORS,width=.67)
        for x,y in enumerate(values):ax.text(x,y+.025,f'{y:.4f}',ha='center',fontsize=10)
        ax.set_xticks(np.arange(4),['Training\nmean','CF','CF +\nmetadata','CF + metadata\n+ time']);ax.set_ylim(0,1.4);ax.set_title(title);ax.grid(axis='y',alpha=.18)
    axs[0].set_ylabel('RMSE (rating points, 1-5 scale; lower is better)')
    return finish(fig,'model_comparison')

def time_chart():
    d=pd.read_csv(ROOT/'results/monthly_activity.csv');dates=pd.to_datetime(d.month)
    fig,axs=plt.subplots(2,1,figsize=(11.8,5.3),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
    axs[0].plot(dates,d.ratings,color='#277b83',marker='o',markersize=3);axs[0].set_yscale('log');axs[0].set_ylabel('Ratings per month\n(logarithmic scale)');axs[0].set_title('Rating-entry activity is concentrated in 2000; later months are much smaller')
    axs[1].plot(dates,d.mean_rating,color='#be4939',marker='o',markersize=3);axs[1].set_ylim(2.8,4.1);axs[1].set_ylabel('Mean observed rating');axs[1].set_xlabel('Rating timestamp month (UTC)')
    for ax in axs:
        ax.grid(alpha=.18);ax.axvline(pd.Timestamp('2000-12-02'),color='#888',ls='--',lw=1);ax.axvline(pd.Timestamp('2000-12-29'),color='#be4939',ls='--',lw=1)
    axs[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4));axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    return finish(fig,'monthly_time')

def gap_chart():
    d=pd.read_csv(ROOT/'results/adjacent_time_gaps.csv')
    fig,ax=plt.subplots(figsize=(11.8,3.8));ax.barh(d.label,d.fraction*100,color='#277b83');ax.invert_yaxis();ax.set_xlim(0,60)
    for i,v in enumerate(d.fraction*100):ax.text(v+.6,i,f'{v:.2f}%',va='center')
    ax.set_xlabel('Share of adjacent within-user rating entries (%)');ax.set_title('53.21% of adjacent entries share the exact same second');ax.grid(axis='x',alpha=.18)
    return finish(fig,'timestamp_ties')

def population_chart():
    d=pd.read_csv(ROOT/'results/temporal_population.csv').iloc[1:];warm=d.warm_fraction.to_numpy()*100
    fig,ax=plt.subplots(figsize=(9,3.5));ax.barh(d.split,warm,color='#277b83',label='Both IDs seen in fitted history');ax.barh(d.split,100-warm,left=warm,color='#ed9753',label='Unknown user or movie')
    for i,v in enumerate(warm):ax.text(v/2,i,f'{v:.2f}% warm',ha='center',va='center',color='white' if v>50 else 'black')
    ax.set_xlim(0,100);ax.set_xlabel('Share of held-out ratings (%)');ax.set_title('Validation and test contain different user cohorts');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False)
    return finish(fig,'warm_cold_population')

def interval_chart():
    import json
    d=json.loads((ROOT/'results/timestamp_experiment.json').read_text())['intervals']
    fig,ax=plt.subplots(figsize=(9,3.5))
    for y,key in enumerate(['random','temporal']):
        val=d[key]['CF + metadata'];mid=val['improvement_RMSE'];low,high=val['ci95']
        ax.errorbar(mid,y,xerr=np.array([[mid-low],[high-mid]]),fmt='o',color='#277b83',capsize=5,markersize=8)
        ax.text(high+.001,y,f'{mid:.4f}',va='center')
    ax.axvline(0,color='#999',lw=1);ax.set_yticks([0,1],['Random test','Future test']);ax.invert_yaxis();ax.set_xlim(-.003,.060);ax.set_xlabel('RMSE without time - RMSE with time (rating points)');ax.set_title('Timestamp increment: paired user-bootstrap 95% intervals');ax.grid(axis='x',alpha=.18)
    return finish(fig,'timestamp_increment')

def genre_chart():
    d=pd.read_csv(ROOT/'results/genre_factor_heatmap.csv',index_col=0)
    fig,ax=plt.subplots(figsize=(12.3,4.8));im=ax.imshow(d,cmap='RdYlGn',vmin=-5,vmax=5,aspect='auto')
    ax.set_xticks(np.arange(d.shape[1]),d.columns,rotation=55,ha='right');ax.set_yticks(np.arange(len(d)),d.index);ax.set_title('Instructor genre-factor characterization, reproduced on the new fit')
    fig.colorbar(im,ax=ax,label='Row-normalized factor loading (arbitrary orientation)',fraction=.025,pad=.02)
    return finish(fig,'genre_factors')

def cv_chart():
    old=pd.read_csv(ROOT/'results/teacher_cv_recomputed.csv');fresh=pd.read_csv(ROOT/'results/fresh_cv_metrics.csv')
    fig,ax=plt.subplots(figsize=(10,3.8));ax.plot(old.archetypes,old.RMSE,'o-',color='#be4939',label='Instructor cached OOF, independently scored');ax.plot(fresh['rank'],fresh.RMSE,'s--',color='#277b83',label='Fresh 400 R fits')
    ax.set_xticks(range(1,21));ax.set_xlabel('Latent rank (1-20)');ax.set_ylabel('20-fold pooled RMSE (rating points)');ax.set_title('Rank selection stays within the 99% training partition');ax.grid(alpha=.18);ax.legend(frameon=False)
    return finish(fig,'cv_curve')

def forecast_chart():
    d=pd.read_csv(ROOT/'results/temporal_test_by_month.csv');d=d[d.rows>=200]
    fig,ax=plt.subplots(figsize=(11.8,3.8))
    for model,c in zip(MODELS,COLORS):
        q=d[d.model==model];ax.plot(pd.to_datetime(q.period),q.RMSE,'o-',color=c,label=model,markersize=3)
    ax.set_ylim(0,1.8);ax.set_ylabel('Monthly RMSE (rating points)');ax.set_xlabel('Future-test rating month (UTC; months with at least 200 ratings)');ax.set_title('Performance varies across future months; users and rating volume also change');ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'));ax.grid(alpha=.18);ax.legend(frameon=False,ncol=2,fontsize=9)
    return finish(fig,'future_monthly_rmse')

def all_charts():
    for fn in [model_chart,time_chart,gap_chart,population_chart,interval_chart,genre_chart,forecast_chart,cv_chart]:
        fig=fn();plt.close(fig)

if __name__=='__main__':all_charts()
