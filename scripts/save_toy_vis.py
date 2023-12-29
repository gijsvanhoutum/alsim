import os
import random
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from aml.models import FastLR
from aml.environment import Environment
from aml.metrics import f1_macro
from aml.strategy2 import RND,US,WUS,AWUS,BEE,EGA,UDD,AWUS2,AWUS3

from sklearn import datasets
import time

def set_priority(pid=None,priority=1):
    """ Set The Priority of a Windows Process.  Priority is a value between 0-5 where
        2 is normal priority.  Default sets the priority of the current
        python process but can take any valid process ID. """
        
    import win32api,win32process,win32con
    
    priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                       win32process.BELOW_NORMAL_PRIORITY_CLASS,
                       win32process.NORMAL_PRIORITY_CLASS,
                       win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                       win32process.HIGH_PRIORITY_CLASS,
                       win32process.REALTIME_PRIORITY_CLASS]
    if pid == None:
        pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, priorityclasses[priority])
    
def get_toy_dataset(n=30):

    x =y  = np.linspace(-1,1,num=n)
    xx,yy = np.meshgrid(x,y)
    X = np.vstack((xx.ravel(),yy.ravel())).T
    
    X3,_ = datasets.make_blobs(n_samples=[n*n],n_features=2)
    X3 = X3 - X3.mean(axis=0)
    X3 = X3 / np.abs(X3).max(axis=0)
    X3 *= 0.25
    X3[:,1] +=0.5

    X = np.vstack((X,X3))
    y = (X[:,1]>-0.5).astype(int)
    
    return {"X":X,"y":y}

def run_repeated_al(axs,environment, agent, iters, reps,n_batch):

    n = 30
    scores = np.zeros((reps,iters))
    
    bounds = np.zeros((reps,iters,n))

    #decision boundary
    xd = np.linspace(-1,1,n)

    
    labeled = []
    yds = []
    
    times = []
    for r in range( reps ):
        random.seed(r) 
        np.random.seed(r)
        start = time.process_time()
        for j in range( iters ):
            
            if j==0:
                state = environment.reset()  
                agent.reset( state )
            else:
                actions = agent.act( state,number=n_batch)
                state = environment.step( actions ) 
            
            clf = state._model
            b = clf.intercept_[0]
            w1, w2 = clf.coef_.T
            c = -b/w2
            m = -w1/w2
            yd = m * xd + c
            bounds[r,j] = yd
            # score on all data
            scores[r,j] = environment.qualities()["quality"][-1]
            
            labeled.append(np.copy(state.known))
            yds.append(yd)
        
        times.append(time.process_time()-start)
        
    q_scores = np.zeros((2,iters))
    q_scores[0] = scores.min(axis=0)
    q_scores[1] = scores.std(axis=0)#np.quantile(scores,0.95,axis=0)
    
    q_bounds = np.zeros((2,iters,n))
    q_bounds[0] = np.quantile(bounds,0.05,axis=0)
    q_bounds[1] = np.quantile(bounds,0.95,axis=0)
    
    for i in range(iters):
        # plot all unlabeled
        x,y,z = environment.X_trn[:, 0],environment.X_trn[:, 1],environment.y_trn
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].get_xaxis().set_ticks([])
        axs[i].get_yaxis().set_ticks([])

        axs[i].scatter(x=x, 
                       y=y, 
                       c=z,
                       s=5,
                       alpha=0.1
                       )
        
        # set title made up performance of classifier 
        m = str(np.round(q_scores[0,i]*100,2))
        axs[i].set_title(m,fontsize=8,pad=2)
    

        # # plot labeled
        axs[i].scatter(x=x[labeled[i]], 
                       y=y[labeled[i]], 
                       c=z[labeled[i]],
                       s=10,
                       alpha=1,
                       edgecolor="black",
                       linewidths=0.5) 
        
        # # plot classifier decision boundary
        axs[i].plot(xd, yds[i], 'g', lw=2, ls='-')
        
        # plot decision boundary range of repeated measurements
        axs[i].plot(xd, q_bounds[0,i], 'r', lw=2, ls='-')
        axs[i].plot(xd, q_bounds[1,i], 'r', lw=2, ls='-')
        
        axs[i].set_ylim([-1,1])
        axs[i].set_xlim([-1,1])
    
    return bounds,q_scores,times

def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch
    
if __name__=="__main__":
    set_priority(priority=5)
    save_dir = os.getcwd()+r"\data\visualization"
    save_dir = r"C:\Users\inst"
    dss = get_toy_dataset()
    clf = partial(FastLR,solver_type=2,tol=1e-4)
    env = Environment(clf,dss,quality=f1_macro,batch=True,stop=2,ratio=-1)
    
    ags = {
        "rnd": RND(),
        "us": US(),
        "wus": WUS(),
        "bee": BEE(),
        "ega": EGA(),
        "udd": UDD(),
        "awus-c": AWUS2(),
        "awus-r": AWUS3(),
            } 
    
    iters = 7
    factor = 2
    reps = 10
    n_batch = 64
    
    width = 190
    height = width*1.1
    figsize = mm2inch(width,height)
    
    fig,ax = plt.subplots(ncols=iters,nrows=len(ags),
                          figsize=figsize,
                          constrained_layout=True,sharey=True,sharex=True)
    
    for i,(name,agent) in enumerate( ags.items() ):  
        print(i,name)

        start = time.process_time()
        b,q,t = run_repeated_al(ax[i],env, agent, iters, reps,n_batch)
        div = round(np.mean(t)*1000,1)
        label = "{}\n({} ms)".format(name,div)
        ax[i,0].set_ylabel(label,rotation=0,labelpad=20,fontsize=8)
        
    ax[-1][0].set_xlabel("Initial: $2$ Labeled",fontsize=8)
    for i in range(iters-1):
        if i==0: 
            ins = "st"
        elif i==1:
            ins = "nd"
        elif i==2:
            ins = "rd"
        else:
            ins = "th"
            
        l = n_batch*(i+1)+2
        p = np.rint(l/len(dss["y"])*100)
        ax[-1][i+1].set_xlabel("$"+str(i+1)+"^{"+ins+"}$ it: "+str(l)+" ("+str(int(p))+"%)",fontsize=8)

    plt.show()
    
    time.sleep(5)
    
    fig.savefig(save_dir+r"\fig4.pdf",
                dpi=300,
                bbox_inches='tight',
                format="pdf",     
            )  
    
    
    
    
    time.sleep(5)