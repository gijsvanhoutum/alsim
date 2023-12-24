import numpy as np
import random
from itertools import product
from functools import partial
from collections import defaultdict
import time
from tqdm import tqdm

def DD(number,d): 
    for n in range(number):
        d = partial(defaultdict,d)
    
    return d()

def run_single(environment, agent, n_batch, n_max): 
    state = environment.reset()   
    agent.reset( state )
    while True: 
        actions = agent.act( state,number=n_batch)
        state = environment.step( actions )      
        if state.done: 
            break

def run_multi(i,env,models,data,quality,agents,batch,n_max):
    epq = DD(4,list)
    for b,m,a in product(batch,models,agents):
        np.random.seed(i)
        random.seed(i)     
        environment = env(models[m],data,quality)
        agent = a()
        run_single(environment,agent,b,n_max)
        qlt = environment.qualities()
        epq[b]["quality"][m][agent.name] =  qlt["quality"]
        epq[b]["known"] = qlt["known"]
         
    return epq

def run_multi_print(i,env,models,data,quality,agents,batch,n_max):
    epq = DD(4,list)
    for b,m,a in tqdm(product(batch,models,agents)):
        np.random.seed(i)
        random.seed(i)     
        environment = env(models[m],data,quality)
        agent = a()
        run_single(environment,agent,b,n_max)
        qlt = environment.qualities()
        epq[b]["quality"][m][agent.name] =  qlt["quality"]
        epq[b]["known"] = qlt["known"]
         
    return epq

def run_to_stop(i,env,models,data,quality,agents,batch,n_max):
    epq = DD(3,list)
    for b,m,a in product(batch,models,agents):
        np.random.seed(i)
        random.seed(i)     
        environment = env(models[m],data,quality)
        agent = a()
        run_single(environment,agent,b,n_max)
        epq[b][m][agent.name] = environment.n_to_stop() 
    return epq

def run_multi2(i,env,models,data,quality,agents,batch,n_max):
    epq = DD(3,list)
    for b,m,a in product(batch,models,agents):
        np.random.seed(i)
        random.seed(i)     
        environment = env(models[m],data,quality)
        agent = a()
        run_single(environment,agent,b,n_max)
        epq[b][m][agent.name] =  environment.qualities()
         
    return epq

def profile_agent(environment,agent,num_calls=1000,batch_sizes=[1]):
    state = environment.reset()   
    out = {}
    # profile reset
    total = []
    for i in range(num_calls):
        t1 = time.time()
        agent.reset( state )
        total.append(time.time()-t1)
        
    reset = np.array([np.mean(total),np.std(total)])
    
    # profile act for different batch sizes
    act = []
    for b in batch_sizes:
        total = []
        for i in range(num_calls):
            t1 = time.time()
            agent.act( state,number=b)
            total.append(time.time()-t1)
            
        act.append([np.mean(total),np.std(total)])
        
    return reset,np.array(act)
    