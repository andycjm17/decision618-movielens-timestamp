"""Next-item MovieLens experiment. Reuses the Recommenders NextItNet encoder.

Adaptation: item-only embeddings and full-catalog softmax replace the library's
item/category + candidate MLP wrapper. Causal residual blocks are called directly
from recommenders.models.deeprec.models.sequential.nextitnet.NextItNetModel.
The protocol, all training grids and sensitivity checks are fixed before test.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,time,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from surprise import Dataset,Reader,SVD
import tensorflow as tf
from recommenders.models.deeprec.models.sequential.nextitnet import NextItNetModel

ROOT=Path(__file__).resolve().parents[1]
SEED=42
def save(name,obj):
    (ROOT/'results'/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))

class NextModel:
    def __init__(self,n,items):
        self.n=n;self.graph=tf.Graph()
        with self.graph.as_default():
            tf.compat.v1.set_random_seed(SEED)
            self.x=tf.compat.v1.placeholder(tf.int32,[None,n]);self.y=tf.compat.v1.placeholder(tf.int32,[None,n])
            emb=tf.compat.v1.get_variable('item_embeddings',[items+1,32],initializer=tf.compat.v1.random_normal_initializer(stddev=.05,seed=SEED))
            h=tf.nn.embedding_lookup(emb,self.x)*tf.cast(self.x[:,:,None]>0,tf.float32)
            encoder=object.__new__(NextItNetModel)
            for layer,dilation in enumerate([1,2,4,8,16]):
                h=encoder._nextitnet_residual_block_one(h,dilation,layer,32,3,causal=True,train=True)
                h=h*tf.cast(self.x[:,:,None]>0,tf.float32)
            self.hidden=h  # Exposed for an independent causal-mask audit.
            w=tf.compat.v1.get_variable('output_weights',[32,items],initializer=tf.compat.v1.glorot_uniform_initializer(seed=SEED))
            b=tf.compat.v1.get_variable('output_bias',[items],initializer=tf.zeros_initializer())
            self.scores=tf.matmul(h[:,-1,:],w)+b
            valid=self.y>0;selected=tf.boolean_mask(h,valid);target=tf.boolean_mask(self.y,valid)-1
            logits=tf.matmul(selected,w)+b
            self.loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target,logits=logits))
            self.update=tf.compat.v1.train.AdamOptimizer(.001).minimize(self.loss)
            self.saver=tf.compat.v1.train.Saver();init=tf.compat.v1.global_variables_initializer()
            self.params=int(sum(np.prod(v.shape.as_list()) for v in tf.compat.v1.trainable_variables()))
        self.sess=tf.compat.v1.Session(graph=self.graph,config=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=2))
        self.sess.run(init)
    def fit_epoch(self,x,y,rng):
        order=rng.permutation(len(x));loss=0.;tokens=0
        for start in range(0,len(x),128):
            ix=order[start:start+128];_,l=self.sess.run([self.update,self.loss],{self.x:x[ix],self.y:y[ix]})
            count=int((y[ix]>0).sum());loss+=l*count;tokens+=count
        return loss/tokens
    def predict(self,x):
        return np.vstack([self.sess.run(self.scores,{self.x:x[s:s+256]}) for s in range(0,len(x),256)])
    def close(self):self.sess.close()

def contexts(sequences,n):
    x=np.zeros((len(sequences),n),np.int32)
    for j,seq in enumerate(sequences):
        tail=seq[-n:];x[j,-len(tail):]=tail
    return x

def chunks(sequences,n):
    # Each within-training transition is a target exactly once per epoch.
    # State resets at chunk boundaries, a documented compute-saving approximation.
    pairs=[]
    for s in sequences:
        for start in range(0,len(s)-1,n):
            x=s[start:start+n];y=s[start+1:start+n+1]
            x=x[:len(y)];pairs.append((x,y))
    a=np.zeros((len(pairs),n),np.int32);b=a.copy()
    for i,(x,y) in enumerate(pairs):a[i,-len(x):]=x;b[i,-len(y):]=y
    return a,b

def evaluate(scores,histories,targets,model,users,ids):
    rows=[];recommendations=[];targets=np.asarray(targets)
    for j,history in enumerate(histories):
        if targets[j]<=0:continue
        sc=scores[j].copy();seen=np.asarray(history);seen=seen[seen>0];sc[seen-1]=-np.inf
        assert targets[j] not in set(seen)
        order=np.argsort(-sc,kind='stable');targetrank=int(np.flatnonzero(order==targets[j]-1)[0])+1
        top=order[:10]+1;hit=int(targetrank<=10)
        rows.append({'user_id':int(users[j]),'model':model,'rank':targetrank,'HR@10':hit,'NDCG@10':float(1/np.log2(targetrank+1)) if hit else 0.,'MRR':1/targetrank,'history_length':len(history)})
        recommendations.extend({'user_id':int(users[j]),'model':model,'rank':k+1,'movie_id':int(ids[v-1]),'score':float(sc[v-1]),'is_target':int(v==targets[j])} for k,v in enumerate(top))
    f=pd.DataFrame(rows);rec=pd.DataFrame(recommendations)
    return {'model':model,**f[['HR@10','NDCG@10','MRR']].mean().to_dict(),'users':len(f),'catalog_coverage':rec.movie_id.nunique()/len(ids)},f,rec

class Experiment:
    def __init__(self):
        for d in ['results','models','figures']: (ROOT/d).mkdir(exist_ok=True)
        self.summary={'seed':SEED,'run_utc':pd.Timestamp.now(tz='UTC').isoformat(),'tensorflow':tf.__version__,
        'protocol':'Per-user final item test, penultimate validation. Equal timestamps ordered by seeded random keys. Final refit includes validation items. Full fitted catalog, seen items removed. Target is next rated movie regardless of stars.',
        'adaptation':'Official Recommenders NextItNet causal residual encoder with item-only 32d embeddings and full-softmax output; replaces item/category candidate-MLP wrapper. Training uses nonoverlapping teacher-forced chunks; state resets at chunk boundaries.',
        'grid':{'windows':[10,20,50],'max_epochs':8,'patience':3,'embedding':32,'dilations':[1,2,4,8,16],'kernel_size':3,'batch_size':128,'learning_rate':.001,'itemknn_k':[40,80],'svd_factors':[50,100],'svd_regularization':.1,'svd_epochs':30},
        'selection':'All model choices and epochs use validation full-catalog NDCG@10. Test never tunes parameters.'}

    def load(self):
        raw=ROOT.parent/'data/raw'
        self.r=pd.read_csv(raw/'ratings.dat',sep='::',engine='python',names=['user','item','rating','time'])
        self.movies=pd.read_csv(raw/'movies.dat',sep='::',engine='python',encoding='latin-1',names=['item','title','genres'])
        self.users=pd.read_csv(raw/'users.dat',sep='::',engine='python',names=['user','gender','age','occupation','zip'])
        assert len(self.r)==1000209 and not self.r.duplicated(['user','item']).any()
        assert self.r.rating.between(1,5).all() and not self.r.isna().any().any()
        assert self.r.user.isin(self.users.user).all() and self.r.item.isin(self.movies.item).all()
        self.r['tie_key']=np.random.default_rng(SEED).random(len(self.r))
        self.r=self.r.sort_values(['user','time','tie_key'],kind='stable').reset_index(drop=True)
        self.r['position']=self.r.groupby('user').cumcount();size=self.r.groupby('user').user.transform('size')
        self.r['split']=np.where(self.r.position==size-1,'test',np.where(self.r.position==size-2,'validation','train'))
        self.train=self.r[self.r.split=='train'];self.val=self.r[self.r.split=='validation'];self.test=self.r[self.r.split=='test'];self.finalfit=self.r[self.r.split!='test']
        self.uid=np.array(sorted(self.r.user.unique()))
        self.r.to_csv(ROOT/'results/sequence_split.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        gap=self.r.groupby('user').time.diff().dropna();prior=self.r.groupby('user').time.shift()
        self.strict=(self.test.time>prior.loc[self.test.index]).to_numpy()
        self.summary['data']={'ratings':len(self.r),'users':len(self.uid),'catalog_movies':len(self.movies),'rated_movies':self.r.item.nunique(),
            'train_ratings':len(self.train),'validation_ratings':len(self.val),'test_ratings':len(self.test),'training_transitions':len(self.train)-len(self.uid),
            'equal_timestamp_fraction':float((gap==0).mean()),'within_60s_fraction':float((gap<=60).mean()),'strict_test_users':int(self.strict.sum()),
            'test_target_tied_with_predecessor':int((~self.strict).sum()),'mean_rating':float(self.r.rating.mean()),'sparsity':1-len(self.r)/(len(self.uid)*len(self.movies))}
        save('protocol_before_training.json',self.summary)
        print(self.summary['data'],flush=True)
        self.setup(self.train,self.val)
        return self.summary['data']

    def setup(self,fit,targets):
        self.fit=fit;self.ids=np.array(sorted(fit.item.unique()));self.mapping={i:k+1 for k,i in enumerate(self.ids)}
        self.hist=[g.item.map(self.mapping).fillna(0).to_numpy(dtype=np.int32) for _,g in fit.groupby('user',sort=True)]
        self.target=targets.set_index('user').reindex(self.uid).item.map(self.mapping).fillna(0).to_numpy(dtype=np.int32)
        self.pop=fit.item.value_counts().reindex(self.ids,fill_value=0).to_numpy(dtype=float)

    def knn(self,k):
        ui={u:j for j,u in enumerate(self.uid)}
        a=sparse.csr_matrix((np.ones(len(self.fit)),(self.fit.user.map(ui),self.fit.item.map(self.mapping)-1)),shape=(len(self.uid),len(self.ids)))
        counts=np.asarray(a.sum(axis=0)).ravel();co=(a.T@a).toarray()
        sim=co/np.sqrt(counts[:,None]*counts[None,:]);np.fill_diagonal(sim,0)
        # Keep k nearest neighbors per source item, cosine of binary interactions.
        for row in sim:
            keep=np.argsort(-row,kind='stable')[:k];mask=np.ones(len(row),bool);mask[keep]=False;row[mask]=0
        return np.asarray(a@sparse.csr_matrix(sim)) if False else (a@sparse.csr_matrix(sim)).toarray()

    def svd(self,k):
        ts=Dataset.load_from_df(self.fit[['user','item','rating']],Reader(rating_scale=(1,5))).build_full_trainset()
        model=SVD(n_factors=k,n_epochs=30,reg_all=.1,random_state=SEED).fit(ts)
        uu=np.array([ts.to_inner_uid(u) for u in self.uid]);ii=np.array([ts.to_inner_iid(i) for i in self.ids])
        return model.pu[uu]@model.qi[ii].T+model.bu[uu,None]+model.bi[ii][None,:]+ts.global_mean

    def tune_baselines(self):
        self.valresults=[]
        for k in [40,80]:
            t=time.perf_counter();sc=self.knn(k);result,_,_=evaluate(sc,self.hist,self.target,'item-kNN',self.uid,self.ids)
            result.update(k=k,seconds=time.perf_counter()-t);self.valresults.append(result);print('Validation',result,flush=True)
        for k in [50,100]:
            t=time.perf_counter();sc=self.svd(k);result,_,_=evaluate(sc,self.hist,self.target,'SVD',self.uid,self.ids)
            result.update(k=k,seconds=time.perf_counter()-t);self.valresults.append(result);print('Validation',result,flush=True)
        self.bestknn=max([x for x in self.valresults if x['model']=='item-kNN'],key=lambda x:x['NDCG@10'])['k']
        self.bestsvd=max([x for x in self.valresults if x['model']=='SVD'],key=lambda x:x['NDCG@10'])['k']
        pd.DataFrame(self.valresults).to_csv(ROOT/'results/baseline_validation.csv',index=False)
        return pd.DataFrame(self.valresults)

    def tune_next(self):
        logs=[];self.windowbest=[]
        for n in [10,20,50]:
            x,y=chunks(self.hist,n);assert int((y>0).sum())==self.summary['data']['training_transitions']
            model=NextModel(n,len(self.ids));rng=np.random.default_rng(SEED);best=-1;bestepoch=0;total=0
            for epoch in range(1,9):
                t=time.perf_counter();loss=model.fit_epoch(x,y,rng);secs=time.perf_counter()-t;total+=secs
                scores=model.predict(contexts(self.hist,n));result,_,_=evaluate(scores,self.hist,self.target,'NextItNet',self.uid,self.ids)
                row={'window':n,'epoch':epoch,'loss':loss,'seconds':secs,**result};logs.append(row);print('NextItNet validation',row,flush=True)
                pd.DataFrame(logs).to_csv(ROOT/'results/nextitnet_training.csv',index=False)
                if result['NDCG@10']>best:
                    best=result['NDCG@10'];bestepoch=epoch;model.saver.save(model.sess,str(ROOT/f'models/validation_n{n}'))
                if epoch-bestepoch>=3:break
            self.windowbest.append({'window':n,'epoch':bestepoch,'NDCG@10':best,'training_seconds':total,'parameters':model.params})
            model.close()
        self.best=max(self.windowbest,key=lambda x:x['NDCG@10'])
        self.summary['selection']={'nextitnet':self.best,'itemknn_k':self.bestknn,'svd_factors':self.bestsvd}
        save('selection_before_test.json',self.summary);pd.DataFrame(self.windowbest).to_csv(ROOT/'results/window_validation.csv',index=False)
        return pd.DataFrame(self.windowbest)

    def final(self):
        self.setup(self.finalfit,self.test);self.summary['final_catalog']=len(self.ids);self.summary['unseen_test_targets']=int((self.target==0).sum())
        metrics=[];per=[];recs=[];times={}
        for name,fn in [('Popularity',lambda:np.tile(self.pop,(len(self.uid),1))),('item-kNN',lambda:self.knn(self.bestknn)),('SVD',lambda:self.svd(self.bestsvd))]:
            t=time.perf_counter();sc=fn();times[name]=time.perf_counter()-t
            r,p,c=evaluate(sc,self.hist,self.target,name,self.uid,self.ids);metrics.append(r);per.append(p);recs.append(c);print('Test',r,flush=True)
        n=self.best['window'];x,y=chunks(self.hist,n);model=NextModel(n,len(self.ids));rng=np.random.default_rng(SEED);t=time.perf_counter()
        for epoch in range(1,self.best['epoch']+1):
            loss=model.fit_epoch(x,y,rng);print('Final refit epoch',epoch,'loss',loss,flush=True)
        times['NextItNet']=time.perf_counter()-t;model.saver.save(model.sess,str(ROOT/'models/final'))
        scores=model.predict(contexts(self.hist,n));r,p,c=evaluate(scores,self.hist,self.target,'NextItNet',self.uid,self.ids)
        metrics.append(r);per.append(p);recs.append(c);print('Test',r,flush=True)
        self.metrics=pd.DataFrame(metrics);self.per=pd.concat(per,ignore_index=True);self.recs=pd.concat(recs,ignore_index=True)
        self.metrics.to_csv(ROOT/'results/test_metrics.csv',index=False);self.per.to_csv(ROOT/'results/test_per_user.csv',index=False);self.recs.to_csv(ROOT/'results/test_top10.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        self.summary['metrics']=metrics;self.summary['fit_and_score_seconds']=times
        self.summary['refit']='All models refitted using train + validation items. No test labels included. Popularity/item-kNN treat every rating as an interaction; SVD retains star values.'
        strictids=self.uid[self.strict];strict=self.per[self.per.user_id.isin(strictids)].groupby('model')[['HR@10','NDCG@10','MRR']].mean()
        strict['users']=self.per[self.per.user_id.isin(strictids)].groupby('model').size();strict.to_csv(ROOT/'results/strict_timestamp_metrics.csv')
        self.summary['strict_metrics']=strict.reset_index().to_dict('records')
        # Paired bootstrap across the common users, not across recommendation slots.
        pivot=self.per.pivot(index='user_id',columns='model',values='NDCG@10');rng=np.random.default_rng(SEED);intervals={}
        for base in ['Popularity','item-kNN','SVD']:
            delta=(pivot.NextItNet-pivot[base]).to_numpy();boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)])
            intervals[base]={'difference':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
        self.summary['paired_ndcg_intervals']=intervals
        self.groups=self.per.merge(self.users[['user','gender','age']],left_on='user_id',right_on='user',validate='many_to_one')
        self.groups['activity']=pd.cut(self.groups.history_length,bins=[0,50,200,np.inf],labels=['19-50','51-200','201+'])
        self.groups.groupby(['model','activity'],observed=True).agg(users=('user_id','size'),HR10=('HR@10','mean'),NDCG10=('NDCG@10','mean')).reset_index().to_csv(ROOT/'results/activity_metrics.csv',index=False)
        self.groups.groupby(['model','gender']).agg(users=('user_id','size'),NDCG10=('NDCG@10','mean')).reset_index().to_csv(ROOT/'results/gender_metrics.csv',index=False)
        targetgenres=self.test[['user','item']].merge(self.movies[['item','genres']],on='item').assign(genres=lambda d:d.genres.str.split('|')).explode('genres')
        self.per.merge(targetgenres,left_on='user_id',right_on='user').groupby(['model','genres']).agg(users=('user_id','size'),HR10=('HR@10','mean'),NDCG10=('NDCG@10','mean')).reset_index().to_csv(ROOT/'results/genre_metrics.csv',index=False)
        self.summary['demo_user']=int(self.uid[self.target>0][0]);du=self.summary['demo_user']
        self.recs[(self.recs.user_id==du)&(self.recs.model=='NextItNet')].merge(self.movies,left_on='movie_id',right_on='item').to_csv(ROOT/'results/demo.csv',index=False)
        np.savez_compressed(ROOT/'models/catalog.npz',movie_ids=self.ids,popularity=self.pop,user_ids=self.uid)
        self.finalfit[['user','item','rating','time','tie_key']].to_csv(ROOT/'models/history.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        model.close();save('summary.json',self.summary)
        return self.metrics

    def sensitivity(self):
        rows=[]
        # Fixed trained weights, alternate available context lengths: inference sensitivity.
        for n in [10,20,50]:
            model=NextModel(n,len(self.ids));model.saver.restore(model.sess,str(ROOT/'models/final'))
            sc=model.predict(contexts(self.hist,n));r,_,_=evaluate(sc,self.hist,self.target,f'context_{n}',self.uid,self.ids);rows.append(r);model.close()
        n=self.best['window'];model=NextModel(n,len(self.ids));model.saver.restore(model.sess,str(ROOT/'models/final'))
        # Keep targets fixed and shuffle only equal-timestamp events in historical inputs.
        rng=np.random.default_rng(99);sh=self.finalfit.copy();sh['tie_key']=rng.random(len(sh));sh=sh.sort_values(['user','time','tie_key'])
        histories=[g.item.map(self.mapping).to_numpy(dtype=np.int32) for _,g in sh.groupby('user',sort=True)]
        r,_,_=evaluate(model.predict(contexts(histories,n)),self.hist,self.target,'reshuffled_timestamp_ties',self.uid,self.ids);rows.append(r)
        # Stronger order perturbation with the same selected recent-item set, not retraining.
        z=contexts(self.hist,n);rng=np.random.default_rng(123)
        for row in z:
            mask=row>0;row[mask]=rng.permutation(row[mask])
        r,_,_=evaluate(model.predict(z),self.hist,self.target,'shuffled_recent_order',self.uid,self.ids);rows.append(r);model.close()
        self.summary['sensitivity']=rows;save('summary.json',self.summary);pd.DataFrame(rows).to_csv(ROOT/'results/sensitivity.csv',index=False)
        return pd.DataFrame(rows)

if __name__=='__main__':
    exp=Experiment();exp.load();exp.tune_baselines();exp.tune_next();exp.final();exp.sensitivity()
