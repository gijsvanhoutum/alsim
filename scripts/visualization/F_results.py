import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch

def plot(ax,df,ticks,fs,name=True):
  
    if name==False:
        anchor = (0.4,0.1,1,1)
    else:
        anchor = (-0.075,0.01,1,1)
        
    axi = inset_axes(ax,width="60%", height="50%",loc="lower right",
                         bbox_to_anchor=anchor,
                         bbox_transform=ax.transAxes)
        
    AUCC = []
    AGES = []
    HHHH = []
    CS = []
    step=0.01
    rng = np.arange(0,1+step,step)
    for i, (age,dfa) in enumerate (df.groupby("agent") ):
        
        H, bins = np.histogram(dfa["percentage"], bins=rng)
        H = np.cumsum(np.insert(H,0,0))
        H = H/H.max() 
        aucc = np.trapz(H)/len(rng) 
        AUCC.append(aucc)
        AGES.append(age)
        HHHH.append(H)
        CS.append(i)
        
    args = np.argsort(AUCC)
    
    for i in args:
        
        age,H,aucc = AGES[i],HHHH[i],AUCC[i]
        color = "C{}".format( CS[i] )
        ax.plot(rng,H,label=age,linewidth=1,zorder=100-i,color=color)
        lab = axi.bar(age,aucc*100,width=0.8,color=color,label=age)
        axi.bar_label(lab, padding=3,fontsize=fs,fmt="%.1f",rotation=90)

        

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_bounds(0,1)  
    ax.spines['bottom'].set_bounds(0,1)   
    ax.yaxis.tick_right() 
    ax.yaxis.set_label_position("right")
    ax.set_xticks(ticks)
    ax.set_xticklabels([int(t*100) for t in ticks],fontsize=fs-2)
    ax.set_yticks(ticks)
    ax.set_yticklabels([int(t*100) for t in ticks],fontsize=fs-2)
    ax.set_ylim([-0.05,1.05])
    ax.set_xlim([-0.05,1.05])

        
    ax.tick_params('both', length=2, width=0.5,top=False,left=False,right=True)
    

    
    axi.spines['top'].set_visible(False)
    axi.spines['right'].set_visible(True)
    axi.spines['bottom'].set_visible(True)
    axi.spines['bottom'].set_linewidth(0.5)
    axi.spines['left'].set_visible(False)
    
    
    axi.set_ylim([55,85])
    axi.set_xticks([])
    axi.set_yticks([])
    # axi.set_yticklabels([int(min(AUCC)*100),int(max(AUCC)*100)],fontsize=fs)  
    # axi.set_yticklabels([int(min(AUCC)*100),int(max(AUCC)*100)],fontsize=fs) 
    axi.set_ylabel("AUCC %",rotation=90,fontsize=fs,labelpad=4)
    axi.yaxis.set_label_position("right")
    axi.tick_params('both', length=1, width=0.5)
    axi.patch.set_alpha(0)
    axi.spines['right'].set_bounds(55,85)  
    return axi
    
def save(fig,path,dpi=300):
    fig.savefig(path,
                dpi=dpi,
                bbox_inches='tight',
                format="pdf",     
            )  
    
if __name__=="__main__":
    cwd = os.getcwd()
    
    save_dir = "../data/visualization/"  
    data_path = "../data/simulation/results.pkl"  

    

    N=2
    fs=8
    xlabel = "Annotated % of training dataset to reach BENCH"
    ylabel = "% of simulations"
    df = pd.read_pickle(data_path)

    df["percentage"] = df["k"] / df["kmx"]
    n_clf = len(df["classifier"].unique())
    n_btc = len(df["batch"].unique())
                
    ticks = np.arange(0,1.1,0.1).round(decimals=1)
    labels = sorted(df["agent"].unique())
    
    lookup_2 = {"RF":"Random forest",
                "GNB":"Gaussian naive bayes",
                "LR":"Logistic regression",
                "SVM":"Support vector machine"}

    ##########################################################################
    # Plot OVERALL
    
    figsize = mm2inch(90,80)
    gridspec = {"hspace":0.05,"wspace":0.025,
                "top":0.99,"left":0,
                "right":0.9,"bottom":0.18}
    
    fig,axes = plt.subplots(ncols=1,
                            nrows=1,
                            figsize=figsize,
                            gridspec_kw=gridspec,
                            sharex=True,
                            )
    
    axi = plot(axes,df,ticks,fs)
    
    axes.set_ylabel(ylabel,rotation=90,ha="center",va="center",fontsize=fs-2)#,labelpad=25)
    axes.set_xlabel(xlabel,rotation=0,ha="center",va="center",fontsize=fs-2,labelpad=5)

    handles,labels = axi.get_legend_handles_labels()
    
    fig.legend(handles,labels, loc="lower center",
                frameon=False,ncol=len(labels),fontsize=fs,columnspacing=0.5,handletextpad=0.25,handlelength=0.75)

    save(fig,save_dir+"F_overall.pdf")
 
    
    ##########################################################################
    # Plot PER BATCH and CLASSIFIER
    
    figsize = mm2inch(190,180)
    gridspec = {"hspace":0.05,"wspace":0.025,
                "top":0.97,"left":0.05,
                "right":0.95,"bottom":0.075}
    
    fig,axes = plt.subplots(ncols=n_clf,
                            nrows=n_btc,
                            figsize=figsize,
                            gridspec_kw=gridspec,
                            sharex=True,
                            )
    

    for i, (clf,dfc) in enumerate( df.groupby("classifier") ):
        axes[0,i].set_title(lookup_2[clf],fontsize=fs)
        for j, (btc,dfb) in enumerate( dfc.groupby("batch") ):
            plot(axes[j,i],dfb,ticks,fs)

            if i==0:
                axes[j,i].yaxis.set_label_position("left")
                axes[j,i].set_ylabel(btc,fontsize=fs,rotation=0,ha="center",va="center",labelpad=15)
    
    handles,labels = axi.get_legend_handles_labels()
    
    fig.legend(handles,labels, loc="lower center",
                frameon=False,ncol=len(labels),fontsize=fs)
    
    fig.text(0.995,0.525,ylabel,rotation=90,ha="center",va="center",fontsize=fs-2)
    fig.text(0.5,0.04,xlabel,rotation=0,ha="center",va="center",fontsize=fs-2)
    fig.text(0.025,0.525,"Batch\nsize",rotation=0,ha="center",va="center",fontsize=fs)
    
    save(fig,save_dir+"F_batch.pdf")
    
    # ##########################################################################
    # # Plot PER DATASET and CLASSIFIER

    
    figsize = mm2inch(190,100)
    gridspec = {"hspace":0.05,"wspace":0.025,
                "top":0.95,"left":0.05,
                "right":0.95,"bottom":0.14}
    
    fig,axes = plt.subplots(ncols=n_clf,
                            nrows=2,
                            figsize=figsize,
                            gridspec_kw=gridspec,
                            sharex=True,
                            )
    


    mask = df["dataset"].str.startswith('DED', na=False)
                              
    for i,m in enumerate([mask,~mask]):

        for j, (clf,dfc) in enumerate( df.loc[m].groupby("classifier") ):
            plot(axes[i,j],dfc,ticks,fs)
            if i==0:
                axes[0,j].set_title(lookup_2[clf],fontsize=fs)    

        axes[i,0].yaxis.set_label_position("left")
        
        if i==0:
            axes[i,0].set_ylabel("DED",fontsize=fs,rotation=0,ha="center",va="center",labelpad=10)
        elif i==1:
            axes[i,0].set_ylabel("OpenML",fontsize=fs,rotation=0,ha="center",va="center",labelpad=10)
                        
    fig.text(0.995,0.55,ylabel,rotation=90,ha="center",va="center",fontsize=fs-2)
    fig.text(0.5,0.075,xlabel,rotation=0,ha="center",va="center",fontsize=fs-2)

         
    handles,labels = axi.get_legend_handles_labels()
    fig.legend(handles,labels, loc="lower center",
                frameon=False,ncol=len(labels),fontsize=fs)

    save(fig,save_dir+"F_dataset.pdf")  

    