import os
import random
import multiprocessing 
from functools import partial
from tqdm import tqdm
import pandas as pd
import sklearn
import numpy as np

from alsim.environment import Environment
from alsim.metrics import f1_macro
from alsim.dataset import save_load_dataset,OpenmlDataset
from alsim.models import FastLR,FastRF,FastGNB
from alsim.strategy import RND,US,WUS,BEE,EGA,UDD,AWUS_C,AWUS_R
from alsim.helpers import run_multi,arg_grid
from alsim.visualization import save_visualization

if __name__=="__main__":
    # Validation for finiteness will be skipped, saving time, potential crashes. 
    sklearn.set_config(assume_finite=True)
    # ensure the same random settings for each simulation
    np.random.seed(0)
    random.seed(0)   
    # current working directory
    cwd = os.getcwd()
    
    # nr of active learning simulation trails with different random init
    nr_sim = 10
    # nr of annotations per iteration (batch)
    batch_sizes = [1,4,16,64] 
    # ratio of dataset data instances to be used for the training dataset
    trn_ratio = 0.5
    # parallel computation method
    parallel = "async"
    # number of CPU to use for parallel
    nr_prc = multiprocessing.cpu_count()
    # save file path
    sim_save_path = cwd+"/data/simulation/results"
    viz_save_dir = cwd+"/data/visualization/"
    # save and load dataset. Create more if needed
    dss = save_load_dataset(
        OpenmlDataset,
        cwd+"/data/source/openml/openml_28.txt",
        cwd+"/data/compressed/openml"
    )
    # if more than 1 dataset, combine them
    # dss = {**dss1,**dss2}
    
    # Machine learning models. All Scikit-learn models are accepted. 
    # FastFit models are custom models optimized
    mls = {
        "LR":partial(FastLR,solver_type=0),
        "GNB":FastGNB,
        "RF":partial(FastRF,n_estimators=10),
        "SVM":partial(FastLR,solver_type=2)
    }
    # active learning query strategies
    ags = {
        "udd": UDD,
        "ega": EGA,
        "bee": BEE,
        "awus-r": AWUS_R,
        "awus-c": AWUS_C,
        "wus": WUS,
        "us": US,
        "rnd": RND
    }
    # active learning environment
    env = partial(Environment,quality=f1_macro,batch=False,stop=1.0)
    
    # all simulation combinations
    args = arg_grid(
        dss,
        mls,
        ags,
        env,
        batch_sizes,
        nr_sim,
        trn_ratio
    )

    # type of execution
    if parallel=="async":
        p = multiprocessing.Pool( nr_prc )  
        pbar = tqdm(total=nr_sim*len(batch_sizes)*len(dss))
        cb = lambda _: pbar.update(1)
        res = [p.apply_async(run_multi, args=i, callback=cb) for i in args] 
        result = [r.get() for r in res]
    elif parallel=="map":
        p = multiprocessing.Pool( nr_prc )  
        result = p.starmap(run_multi,args)    
    else:
        result = [run_multi(*i) for i in tqdm(args)]
        
    # save results as dataframe
    b = [item for sublist in result for item in sublist]
    cols = ["batch","classifier","agent","dataset","size","k","q","kmx","qmx"]
    df = pd.DataFrame(b,columns=cols)
    df.to_pickle(sim_save_path,protocol=3)
    
    # save results as figures
    save_visualization(viz_save_dir,sim_save_path)