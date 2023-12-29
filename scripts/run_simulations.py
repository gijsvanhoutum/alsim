# standard library
import os
import random
import multiprocessing 
from functools import partial

# externel library
from tqdm import tqdm
import pandas as pd
import sklearn
import numpy as np

# internal library
from aml.environment import Environment
from aml.metrics import f1_macro
from aml.dataset import load_dataset
from aml.models import FastLR,FastRF,FastGNB
from aml.strategy2 import RND,US,WUS,AWUS,BEE,EGA,UDD,AWUS3,AWUS2
from aml.helpers import run_multi,arg_grid

if __name__=="__main__":

    np.random.seed(0)
    random.seed(0)   
    sklearn.set_config(assume_finite=True)
    
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'
    save = data_dir+r'\simulation\results.pkl'
    parallel = "map"
    nr_prc = multiprocessing.cpu_count()
    nr_sim = 1000
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
        "udd": UDD,
        "ega": EGA,
        "bee": BEE,
        "awus-r": AWUS3,
        "awus-c": AWUS2,
        # "awus-old": AWUS,
        "wus": WUS,
        "us": US,
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
