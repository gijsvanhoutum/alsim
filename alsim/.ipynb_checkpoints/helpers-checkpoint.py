import numpy as np
import random
from itertools import product

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