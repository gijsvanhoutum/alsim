import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])

import numpy as np

import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from functools import partial

from aml.models import FastLR,FastRF,FastNB
import multiprocessing
import pandas as pd
import itertools


def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch
    
class XirisDataset:

    @classmethod
    def _compress_frame(cls,frame, dim,return_all=False):
        frame = frame.astype(np.float32)
        frame = cls.min_max_scale(frame)
        hgm = cls._histogram_of_gradient_magnitudes( frame, dim )
        hcm = cls._histogram_of_intensity_magnitudes( frame, dim )
        hist = np.hstack( (hgm,hcm))
        return hist.astype(np.uint64)
    
    @classmethod
    def log_norm_transform(cls,hist):      
        return np.log( hist+ 1 ) / np.log( hist.sum() / 2 + 1 ) -0.5
    
    @classmethod
    def min_max_scale(cls,frame):
        mn,mx = frame.min(),frame.max()
        if mx==mn:
            return np.zeros(frame.shape,dtype=np.float32)
        else:
            return ( frame - mn ) / (mx-mn)
        
    @classmethod
    def _histogram_of_gradient_magnitudes(cls, frame, dim, kernel=(3,3)):
        sobelx = cv2.Sobel( frame, cv2.CV_32F,1,0,ksize=3)  
        sobely = cv2.Sobel( frame,cv2.CV_32F,0,1,ksize=3)      
        mag = cv2.magnitude(sobelx, sobely) / np.sqrt(2)
        hist = cv2.calcHist([mag],[0],None,[dim],[0,1]).ravel()
        return hist

    @classmethod
    def _histogram_of_intensity_magnitudes(cls, frame, dim ):
        hist = cv2.calcHist([frame],[0],None,[dim],[0,1]).ravel()
        return hist
        
    @classmethod
    def get_datasets(cls, source_path, dim):
        d = {}
        for video_name in tqdm( os.listdir( source_path ) ):   
            video_path = os.path.join( source_path, video_name)
            
            features,labels = [],[]
            for class_name in os.listdir( video_path ):
                class_path = os.path.join( video_path, class_name)     
                for frame_name in os.listdir( class_path ):
                    frame_path = os.path.join( class_path, frame_name)
                    PIL_image = Image.open(frame_path)
                    feature = np.array(PIL_image).astype(np.uint8)
                    feature = cls._compress_frame( feature,dim )
                    features.append(feature)
                    labels.append(class_name)
            
            f,l = np.array(features),np.array([labels]).T
            d[video_name] = np.hstack((f,l))
        
        return d  
    
    @classmethod
    def preprocess(cls, dataset, ng=0,ni=0):
        out = {}

            
        for k,v in dataset.items():
            vecs,lbls = v[:,:-1].astype(np.uint64),v[:,-1][:,None]
            s = vecs.shape[1]//2

            if ng>1:
                gv = np.split( vecs[:,:s],ng,axis=1)
                gv = np.hstack([v.sum(axis=1)[:,None] for v in gv])
            elif ng==1:
                gv = vecs[:,:s].sum(axis=1)[:,None]

            if ni>1:
                iv = np.split( vecs[:,s:],ni,axis=1)
                iv = np.hstack([v.sum(axis=1)[:,None] for v in iv])
            elif ni==1:
                iv = vecs[:,s:].sum(axis=1)[:,None]

            if ng>0 and ni>0:
                feat = np.hstack((gv,iv)).astype(np.float64)
            elif ng>0:
                feat = gv.astype(np.float64)
            elif ni>0:
                feat = iv.astype(np.float64)
            else:
                feat = vecs.astype(np.float64)
                
            for i in np.arange(feat.shape[0]):
                feat[i] = cls.log_norm_transform(feat[i])

            out[k] = np.hstack((feat,lbls))
            
        return out
            
    @classmethod
    def load_dataset(cls, save_path ):
        npz = np.load( save_path,allow_pickle=True)      
        return {key:npz[key] for key in npz.files}
    
    @classmethod
    def save_dataset(cls, source_path, save_path,dim ):
        datasets = cls.get_datasets( source_path,dim )
        np.savez(save_path,**datasets)

    @classmethod
    def split(cls,dataset,ratio):
        out = {}
        for k,v in dataset.items():
            lbls = v[:,-1]
            nr = len(lbls)
            indices = np.arange(nr)
            
            train = np.zeros(nr).astype(np.bool)
            for l in np.unique(lbls):
                rng = indices[lbls==l]
                inds = np.random.choice(rng,size=int(len(rng)*ratio),replace=False)
                train[inds] = True
                
            out[k] = {"train": v[train],"test":v[~train]}
                
            
        return out

def run_diag(i,complete,n_feats,clfs):
    
    np.random.seed(i)
    split = XirisDataset.split(complete,0.5)
    
    out = []
    for i,f in enumerate( n_feats ):
        
        for d,p in {"100% Intensity":(0,f),"50/50% Grad. Int.":(f//2,f//2),"100% Gradient":(f,0)}.items():
            
            for k,v in split.items():
                v = XirisDataset.preprocess(v,ng=p[0],ni=p[1])
                X_trn = v["train"][:,:-1].astype(np.float64)
                y_trn = v["train"][:,-1].astype(str)
                X_tst = v["test"][:,:-1].astype(np.float64)
                y_tst = v["test"][:,-1].astype(str)           
                
                
                for i, (n,c) in enumerate( clfs.items() ):
                    
                    clf = c()
                    clf.fit(X_trn,y_trn)
                    y_pred = clf.predict(X_tst)
                    sc = f1_score(y_tst, y_pred, average='macro')
                    out.append( (d,p[0]+p[1],k,n,sc))
                   
    return out

def format_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['right'].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)    

def plot(df):
    ms = 3
    num = 10

    fig3,ax = plt.subplots(nrows=2, 
                           ncols=1,
                           figsize=mm2inch(90,90),
                           constrained_layout=True
                           )

    ax[0].set_title('(a)',fontsize=8)
    format_ax(ax[0])
    ax[1].set_title('(b)',fontsize=8)
    format_ax(ax[1])
    
        
    colors = ["C0","C1","C2","C3"]
    markers = ["o","v","s","x"]
    ymin,ymax = 1,0
    for i,(f,dff) in enumerate( df.groupby("f") ):
        
        dffg = dff.groupby("n")
        q5 = dffg.quantile(q=0.25)
        q50 = dffg.quantile(q=0.5)
        q95 = dffg.quantile(q=0.75)
        
        ymin = np.minimum(q5["sc"].min(),ymin)
        ymax = np.maximum(q95["sc"].max(),ymax)
        
        x = np.arange(len(q50))
        
        ax[0].plot(x,q50["sc"],color=colors[i],ls="-",lw=1,marker=markers[i],ms=ms,label=f)
        ax[0].fill_between(x,q5["sc"],q95["sc"],alpha=0.15,color=colors[i])
                
    dataset = XirisDataset.preprocess(complete,ni=8,ng=8)

    total = np.vstack([v for k,v in dataset.items()])
    x = np.arange(total.shape[1]-1)
    for i,c in enumerate( np.unique(total[:,-1]) ):
        
        feat = total[total[:,-1]==c,:-1].astype(float)
        q5 = np.quantile(feat,0.25,axis=0)
        q50 = np.quantile(feat,0.5,axis=0)
        q95 = np.quantile(feat,0.75,axis=0)
        
        ids = np.split(x,2)
        
        print(c)
        label = c.split("_")[0]+" "+c.split("_")[1]
        label = label.capitalize()
        ax[1].plot(x[ids[0]],q50[ids[0]],color=colors[i],ls="-",lw=1,label=label,marker=markers[i],ms=ms)
        
        ax[1].fill_between(x[ids[0]],q5[ids[0]],q95[ids[0]],alpha=0.15,color=colors[i])
        ax[1].plot(x[ids[1]],q50[ids[1]],color=colors[i],ls="-",lw=1,marker=markers[i],ms=ms)
        ax[1].fill_between(x[ids[1]],q5[ids[1]],q95[ids[1]],alpha=0.15,color=colors[i])
    
    ax[0].set_xticks(np.arange(len(np.unique(dff["n"]))))
    ax[0].set_xticklabels(components,fontsize=6)
    
    ax[0].set_yticks(np.linspace(0,1,num=num+1))
    ax[0].set_yticklabels(np.linspace(0,100,num=num+1,dtype=int),fontsize=6)    
    ax[0].set_ylim([ymin,ymax])  
    ax[0].set_xlabel("Nr of features",fontsize=6)
    ax[0].set_ylabel("F1-macro score [%]",fontsize=6)
    leg = ax[0].legend(fontsize=6,
                 frameon=False,
                 loc="lower right",
                 handlelength=1,
                 borderaxespad=0.5,
                 borderpad=0,
                 title="Features",
                 title_fontsize=6)
    
    leg._legend_box.align = "left"
    
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(x,fontsize=6)    
    ax[1].set_ylim([-0.5,0.5])  
    ax[1].set_yticks([-0.5,0,0.5])
    ax[1].set_yticklabels([-0.5,0,0.5],fontsize=6)   
    leg = ax[1].legend(fontsize=6,
                 frameon=False,
                 loc="lower right",
                 handlelength=1,
                 borderaxespad=0.5,
                 borderpad=0,
                 title="Classification",
                 title_fontsize=6)
    
    leg._legend_box.align = "left"
    
    ax[1].set_ylabel("Value",fontsize=6)
    ax[1].tick_params(axis="x", which=u'both',length=0)
    ax[1].set_xticks([3.5,11.5])
    ax[1].set_xticklabels(["Gradient features","Intensity features"],fontsize=6)  
    return fig3


if __name__=="__main__":
    source_path = "../data/source/xiris/"
    save_path = "xiris.npz"
    save_dir = "../data/visualization/"    
    
    if not os.path.exists(save_path):
        XirisDataset.save_dataset(source_path, save_path, dim=256)
    
    complete = XirisDataset.load_dataset(save_path)
    
    components = [4,8,16,32,64,128,256]
    
    clfs = {"GNB": FastNB,
            "LR": partial(FastLR,max_iter=10000),
            "SVM": partial(FastLR,max_iter=10000,solver_type=2),
            "RF": partial( FastRF, n_estimators=10)
            }

    nr_sim = 11
    nr_prc = multiprocessing.cpu_count()-1
    
    p = multiprocessing.Pool( nr_prc )  
    pbar = tqdm(total=nr_sim)
    cb = lambda _: pbar.update(1)

    res = [p.apply_async(run_diag, args=(i,complete,components,clfs), callback=cb) for i in range(nr_sim)] 
    result = [r.get() for r in res]

    result = list(itertools.chain(*result))

    df = pd.DataFrame(result,columns=("f","n","ds","cf","sc"))

    fig = plot(df)

    fig.savefig(save_dir+r'/F_features.pdf',
                format='pdf', 
                dpi=300, 
                bbox_inches='tight')