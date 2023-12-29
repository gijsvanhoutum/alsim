# standard library
import os
import random
import multiprocessing 
from functools import partial
import numpy as np

# externel library
from tqdm import tqdm
import pandas as pd
import sklearn

# internal library
from aml.environment import Environment
from aml.metrics import f1_macro,accuracy_score
from aml.dataset import load_dataset
from aml.models import FastLR,FastRF,FastGNB
from aml.strategy import RND,US,WUS,AWUS,BEE,EGA,UDD
from aml.helpers import run_multi,arg_grid

import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from functools import partial
from sklearn import mixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def indices(y,ratio=0.5): 

    train = np.zeros(y.size,dtype=np.bool)
    for k in np.unique(y):
        mask = (y==k).astype(int)
        nr = int(np.count_nonzero(mask) * ratio)
        p = mask / mask.sum()
        ids = np.random.choice(len(mask),size=nr,replace=False,p=p)
        train[ids] = True    
            
    return train

def single(dataset,clf,metric): 
    
    keys = list(dataset.keys())
    
    tst_r = {}
    for i in keys:
        tst_r[i] = []
        for j in range(100):
            train = indices(dataset[i]["y"],ratio=0.1)
            test = ~train
            trn_X,tst_X = dataset[i]["X"][train],dataset[i]["X"][test]
            trn_y,tst_y = dataset[i]["y"][train],dataset[i]["y"][test]
    
            clf.fit( trn_X , trn_y )
                
            y_pred = clf.predict(tst_X)
            qlt = metric(tst_y,y_pred,labels=np.unique(trn_y)[:,None])
            tst_r[i].append( qlt )

        tst_r[i] = np.array(tst_r[i])
    return tst_r

def multi(dataset,clf,metric): 
    
    keys = list(dataset.keys())
    
    results = {}
    for j in range(100):
        results[j] = []
        X_trn,X_tst,y_trn,y_tst = [],[],[],[]
        
        for i in keys:
            train = indices(dataset[i]["y"],ratio=0.5)
            test = ~train
            X_trn.append( dataset[i]["X"][train] )
            X_tst.append( dataset[i]["X"][test] )
            y_trn.append( dataset[i]["y"][train] )
            y_tst.append( dataset[i]["y"][test] )
        
        X_trn,X_tst = np.vstack(X_trn),np.vstack(X_tst)
        y_trn,y_tst = np.hstack(y_trn),np.hstack(y_tst)
        
        clf.fit( X_trn ,y_trn )
                    
        y_pred = clf.predict(X_tst)
        qlt = metric(y_tst,y_pred,labels=np.unique(y_trn)[:,None])
        results[j].append(qlt)
        
    return results

def split(dataset,clf,metric): 
    
    keys = list(dataset.keys())

    tst_r,trn_r = {},{}
    for i in np.arange(1,len(keys)):

        tst_r[i],trn_r[i] = [],[]
        for trn_c in combinations(keys,i):
            
            
            trn_X = np.vstack([dataset[j]["X"] for j in trn_c ])
            trn_y = np.hstack([dataset[j]["y"].ravel() for j in trn_c ])
            
            clf.fit(trn_X,trn_y)
            
            for k in trn_c:
                y_pred = clf.predict(dataset[k]["X"])
                qlt = metric(dataset[k]["y"],y_pred,labels=np.unique(dataset[k]["y"])[:,None])
                trn_r[i].append(qlt)
                
            for k in keys:
                if k in trn_c:
                    continue
                
                y_pred = clf.predict(dataset[k]["X"])
                qlt = metric(dataset[k]["y"],y_pred,labels=np.unique(dataset[k]["y"])[:,None])
                tst_r[i].append(qlt)

        tst_r[i] = np.array(tst_r[i])
    return tst_r

if __name__=="__main__":

    np.random.seed(0)
    random.seed(0)   
    sklearn.set_config(assume_finite=True)
    
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'
    save = data_dir+r'\simulation\results2.pkl'
    parallel = "async"
    nr_prc = multiprocessing.cpu_count()-1
    nr_sim = 1000
    btcs = [1,4,16,64]
    trn_ratio = 0.5
    comp_dir = data_dir+r"\compressed"

    dss = load_dataset( comp_dir, date=-1, hint="xiris")
    clf = FastRF(n_estimators=100)
    clf = FastLR()
    #clf = mixture.BayesianGaussianMixture(n_components=3)
    #clf = mixture.GaussianMixture(n_components=3,covariance_type="full")
    #clf = LinearDiscriminantAnalysis()
    #clf = MLPClassifier()
    tst_r = multi(dss,clf,f1_macro)
    
    fig,ax = plt.subplots()
    for i,(k,v) in enumerate( tst_r.items() ):
        v = v[~np.isnan(v)]
        ax.violinplot(v,positions=[i],showmeans=True)
        ax.scatter(x=np.ones(len(v))*i-0.05,y=v,s=5,color="black")
        
    ax.set_ylim([0,1])

    # mls = {
    #     "LR":FastLR,
    #     "GNB":FastGNB,
    #     "RF":partial(FastRF,n_estimators=10),
    #     "SVM":partial(FastLR,solver_type=2)
    #         }
    
    # ags = {
    #     "udd": UDD,
    #     "ega": EGA,
    #     "bee": BEE,
    #     "awus": AWUS,
    #     "wus": WUS,
    #     "us": US,
    #     "rnd": RND
    #         }
        
    # env = partial(Environment,quality=f1_macro,batch=False,stop=1.0)
    
    # args = arg_grid(dss,mls,ags,env,btcs,nr_sim,trn_ratio)

    # if parallel=="async":
    #     p = multiprocessing.Pool( nr_prc )  
    #     pbar = tqdm(total=nr_sim*len(btcs)*len(dss))
    #     cb = lambda _: pbar.update(1)
    #     res = [p.apply_async(run_multi, args=i, callback=cb) for i in args] 
    #     result = [r.get() for r in res]
    # elif parallel=="map":
    #     p = multiprocessing.Pool( nr_prc )  
    #     result = p.starmap(run_multi,args)    
    # else:
    #     result = [run_multi(*i) for i in tqdm(args)]
        
    # b = [item for sublist in result for item in sublist]
    # cols = ["batch","classifier","agent","dataset","size","k","q","kmx","qmx"]
    # df = pd.DataFrame(b,columns=cols)

    # if save:
    #     df.to_pickle(save,protocol=3)
