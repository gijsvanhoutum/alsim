import os

import multiprocessing 
from functools import partial
import numpy as np

import sklearn

from aml.environment import Environment
from aml.metrics import f1_macro
from aml.dataset import load_dataset

from aml.models import FastLR,FastRF,FastGNB
from aml.strategy2 import RND,US,WUS,AWUS,BEE,EGA,UDD

from itertools import product
import random
from tqdm import tqdm
import pandas as pd


def run_single(environment, agent, n_batch): 
    state = environment.reset()  
    if state.done:
        return environment.qualities()
    
    agent.reset( state )
    while True: 
        actions = agent.act( state,number=n_batch)
        state = environment.step( actions )      
        if state.done: 
            break
        
    return environment.qualities()

def run_multi(i,env,models,agents,name,dataset,train_ratio,batch_size):
    out = []
    size = len( dataset["y"] )
    for m,a in product( models, agents):
        np.random.seed(i)
        random.seed(i)  
        environment = env(models[m],dataset,ratio=train_ratio)
        e = run_single(environment,agents[a](),batch_size)
        out.append((batch_size,
                    m,
                    a,
                    name,
                    size,
                    e["known"][-2],
                    e["quality"][-2],
                    e["known"][-1],
                    e["quality"][-1])
                   )
         
    return out
    
def arg_grid(datasets,models,agents,environment,batch_sizes,simulations,ratio):
    args = []
    sizes = []
    for n,d in datasets.items():
        for b in batch_sizes:
            for i in range(simulations):
                sizes.append(len(d["y"]))
                args.append( (i,environment,models,agents,n,d,ratio,b) )
        
    inds = np.argsort(sizes)[::-1]
    
    return [args[i] for i in inds]

if __name__=="__main__":

    np.random.seed(0)
    random.seed(0)   
    sklearn.set_config(assume_finite=True)
    
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'
    save = data_dir+r'\simulation\results2.pkl'
    parallel = "async"
    nr_prc = multiprocessing.cpu_count()
    nr_sim = 10
    btcs = [1,4,16,64]
    trn_ratio = 0.5
    comp_dir = data_dir+r"\compressed"

    dss1 = load_dataset( comp_dir, date=-1, hint="openml")
    dss2 = load_dataset( comp_dir, date=-1, hint="xiris")
    dss = {**dss1, **dss2}

    mls = {
        "LR":FastLR,
        "GNB":FastGNB,
        "RF":partial(FastRF,n_estimators=10),
        "SVM":partial(FastLR,solver_type=2)
            }
    
    ags = {
        "udd": partial(UDD,l=1/3,b=1/3,method="max"),
        "ega": partial(EGA,t=0.01,b=0.01,method="max"),
        "bee": partial(BEE,l=0.1,e=0.01,method="max"),
        "awus": partial(AWUS,method="max"),
        "wus": partial(WUS,method="max"),
        "us": partial(US,method="max"),
        "rnd": RND
            }
        
    env = partial(Environment,quality=f1_macro,batch=False,stop=1.0)
    
    args = arg_grid(dss,mls,ags,env,btcs,nr_sim,trn_ratio)

    if parallel=="async":
        p = multiprocessing.Pool( nr_prc )  
        pbar = tqdm(total=nr_sim*len(btcs)*len(dss))
        cb = lambda _: pbar.update(1)
        res = [p.apply_async(run_multi, args=i, callback=cb) for i in args] 
        result = [r.get() for r in res]
    elif parallel=="map":
        p = multiprocessing.Pool( nr_prc )  
        result = p.starmap(run_multi,args)    
    else:
        result = [run_multi(*i) for i in tqdm(args)]
        
    b = [item for sublist in result for item in sublist]
    cols = ["batch","classifier","agent","dataset","size","k","q","kmx","qmx"]
    df = pd.DataFrame(b,columns=cols)

    if save:
        df.to_pickle(save,protocol=3)
