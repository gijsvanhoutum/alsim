import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import cm

from math import atan2,degrees
import numpy as np

#Label line with line2D label data
def labelLine(line,x,label,fs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip]

    ax.text(x,y,label,fontsize=fs,rotation=0,ha="center",va="center",color="black",#line.get_color(),
            bbox=dict(pad=2,facecolor='white', edgecolor='none'))

def labelLines(lines,xvals,labels,fs):

    for i,(line,x,label) in enumerate( zip(lines,xvals,labels) ):

        #if i!=len(labels)-1:
        label = "s = "+str(round(label,2))
        #else:
        #    label = r'a $\approx$ '+str(int(label))

        labelLine(line,x,label,fs)
        

def plot_pmf(ax,es,fs,num=100000):
    # plt.rc('axes', prop_cycle=c_cms)
    u=np.linspace(0,1,num=num)#*0+1
    #u = np.random.rand(num)
    ymax = 0.9/num*2

    n = (u-u.min()) / (u.max() - u.min() )
    for a in es:
        e = 1.0/(1.0-a)-1.0
        p = (n+1)**e
        p/=p.sum()
        
        txt = None
        if a==0:
            txt = "Uniform random sampling (RND)"
            ls = "-"
        elif a==1/2:
            txt = "Proportional weighted\nuncertainty sampling (WUS)"
            ls = "-."
        elif a == es[-1]:
            ls = "--"
            txt = "Max uncertainty sampling (US)"
        
        if not txt:    
            ax.plot(n,p,color="k",ls=(0, (1, 1)),lw=1)
        else:
            ax.plot(n,p,label=txt,lw=2,ls=ls)
            
        idx = np.argmax(p>ymax)-1
        if idx>0:
            du = u[idx]-u[idx-1]
            dp = p[idx]-p[idx-1]
            r = dp/du
            factor = 0.999999
            ax.annotate("",fontsize=fs, 
                        xy=(u[idx],ymax), 
                        xytext=(u[idx]*factor,ymax-(r*(1-factor))),
                        xycoords='data',
                        arrowprops=dict(arrowstyle="->",fc="b",shrinkA=0,shrinkB=0),
                        va="bottom")
        

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position("right") 
    ax.yaxis.set_label_position("right")
    ax.set_xlabel("Instance uncertainty u(x)",fontsize=fs,labelpad=0)
    ax.set_ylabel("Sampling\nprob p(x)",fontsize=fs,rotation=0,ha="center",labelpad=10)

    xt = [0,1]
    ax.set_xticks(xt)
    ax.set_xticklabels(["min","max"],fontsize=fs)  
    ax.legend(fontsize=fs,frameon=False,loc="upper left",handlelength=3,borderaxespad=0,borderpad=0)

    ax.set_yticks([0])
    ax.set_yticklabels([0],fontsize=fs)  

    ax.set_ylim(-0.05/num,ymax)  # most of the data
    ax.set_xlim(0,1.025)  # most of the data
    ax.spines['right'].set_bounds(0,ymax)  
    ax.spines['bottom'].set_bounds(0,1)  

def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch
    
if __name__=="__main__":
    save_dir = "../data/visualization/"
    
    fs = 6


    es = [0,0.5,0.8,0.95,0.999]
    xvals = [0.1,0.225,0.425,0.7,0.985]
    
    figsize = mm2inch(90,60)
    kw = {"hspace":0.5,"wspace":0.05,
          "top":0.95,"left":0.05,"right":0.85,"bottom":0.15}
    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=figsize,gridspec_kw=kw)
    

    plot_pmf(ax,es,fs)
    
    labelLines(plt.gca().get_lines(),xvals,es,fs)

    # # plot_se(ax[1],fs)
    # fig.set_size_inches(width, height)
    plt.savefig(save_dir+'F_exponent.pdf',
                format='pdf', 
                dpi=300, 
                bbox_inches='tight')
    