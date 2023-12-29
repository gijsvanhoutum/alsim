



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import inset_axes 


def calculate(percentages):
    C = np.cumsum(percentages)
    N = C/C.max() 
    A = np.trapz(N)/100
    return C,N,A
            
def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch

def gauss(x,h=1, mid=0,sd=1):
    out = h * np.exp(-pow(x-mid, 2)/(2*sd**2))
    return out / sum(out)

def expon(x,h=1, mid=0,sd=1):
    out = np.exp(-x)
    out[0] = min(out)
    return out / sum(out)
    
def lognorm(x,h=1, mid=0,sd=1):
    return 1.0 / (x*sd*np.sqrt(2*np.pi)) * np.exp( -(np.log(x)-mid+1)**2 / (2*sd**2))
    
def plot_lc(ax,h=0.9):
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position("right") 
    ax.set_xticks([0,10])
    ax.set_xticklabels(["0%","100%"],fontsize=fs-2)
    ax.set_xlim([0,10])
    ax.set_yticks([h])
    ax.set_yticklabels(["BENCH"],fontsize=fs-2)
    ax.spines['right'].set_bounds(0,h)  
    
    #ax.hlines(y=h, xmin=0, xmax=10,ls="--",lw=0.5,color="k")
    x = np.linspace(0,10,num=1000)
    return x,h

def plot_cc(ax,h=0.9):
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_linewidth(0.5)
    ax.yaxis.set_ticks_position("right") 
    ax.set_xticks([0,10])
    ax.set_xticklabels(["0%","100%"],fontsize=fs)
    # ax.set_xlim([0,10])
    ax.set_yticks([0,1])
    ax.set_yticklabels(["0%","100%"],fontsize=fs)
    ax.spines['right'].set_bounds(0,1)  
    ax.spines['bottom'].set_bounds(0,10) 
    
def plot_aucc(ax,n1,n2):
    tot = np.trapz(n1*0+1)
    ax.bar(0,np.trapz(n1)/tot,edgecolor="steelblue",width=0.75)#,color="none")
    ax.bar(1,np.trapz(n2)/tot,edgecolor="orange",width=0.75)#,color="none")

def curve(ax,x,h,l=1,r=3,c1="lightgray",c2="steelblue",func=gauss):
    
    low = 1.0 - np.exp(-l*x)
    low = low / max(low)
    high = 1.0 - np.exp(-r*x)
    high = high / max(high)  
    x_high = x[low<h][-1]
    x_low = x[high<h][-1]
    
    low[low>h]=h
    high[high>h]=h


        
    ax.fill_between(x,low,high,color="lightgray",zorder=5,edgecolor="white")   

    xg = x[(x>x_low) & (x<x_high)]

    yg = func(xg,h=1,mid=0.5*(x_high+x_low),sd=0.2*(x_high-x_low))
    # yg = yg - min(yg) +h

    ygp = 20*(yg - min(yg) ) / sum(yg)  + h

    ax.plot(xg,ygp,lw=1.5,ls="-",color=c2,zorder=20)

    x_ex = np.array([1.0 - np.exp(-((r-l)/(i+3)*np.random.randn()+((r+l)/2))*i) for i in x])
    x_ex = x_ex / max(x_ex)  


    x_ex = np.interp(np.linspace(min(x),max(x),num=len(x)), x[0::30], x_ex[0::30])
    a = x_ex[x_ex>=h][0]
    b = x[x_ex>=h][0]
    ax.scatter(b,a,zorder=30,s=20)
    ax.plot(x[x<b],x_ex[x<b],lw=1,ls="--",color="k",zorder=10)
    

    g = func(x,h=1,mid=0.5*(x_high+x_low),sd=0.2*(x_high-x_low))
    
    g = np.zeros(len(x))
    g[(x>x_low) & (x<x_high)] = yg
    return g,x_low,x_high

if __name__=="__main__":

    save_dir = "../data/visualization/"
    fs=8
    
    figsize = mm2inch(90,90)
    kw = {"hspace":0.5,"wspace":0.05,"height_ratios":[3,3],
          "top":0.95,"left":0.025,"right":0.725,"bottom":0.125}
    
    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=figsize,gridspec_kw=kw)
    
    
    x,h = plot_lc(ax[0])
    
    np.random.seed(2)
        
    ax[0].axhline(h,color="k",ls="-",lw=0.5,zorder=20)
    ax[0].set_xlabel("Annotated % of training dataset\nused for classifier training",fontsize=fs-2,labelpad=-10)
    ax[0].set_ylabel("F1-macro score\non validation\ndataset",fontsize=fs-2,rotation=0,ha="center",labelpad=10)
    ax[0].yaxis.set_label_position("right")
    ax[0].set_title("(a) Active learning curves",fontsize=fs,loc="left")
    y1,l1,h1 = curve(ax[0],x,h,l=0.8,r=3,c1="lightblue",c2="steelblue")
    y2,l2,h2 = curve(ax[0],x,h,l=0.3,r=0.75,c1="wheat",c2="orange")
    # y3,l3,h3 = curve(ax[0],x,h,l=0.3,r=0.5,c1="black",c2="black",func=expon)
    
    for a,i in zip(ax.flatten(),[h,1]):
        a.vlines(x=l1, ymin=0, ymax=i,ls="--",lw=0.5,color="gray")
        a.vlines(x=h1, ymin=0, ymax=i,ls="--",lw=0.5,color="gray")
        a.vlines(x=l2, ymin=0, ymax=i,ls="--",lw=0.5,color="gray")
        a.vlines(x=h2, ymin=0, ymax=i,ls="--",lw=0.5,color="gray")
        
    plot_cc(ax[1])
    
    c1,n1,a1 = calculate(y1)
    c2,n2,a2 = calculate(y2)
    # c3,n3,a3 = calculate(y3)
    
    ax[1].set_title("(b) Cumulative dist. of BENCH crossings",fontsize=fs,loc="left")
    ax[1].plot(x,n1,color="steelblue",ls="-",lw=1.5)
    #ax[1].fill_between(x,0,n1,color="none",hatch="\\\\",edgecolor="lightblue")
    ax[1].plot(x,n2,color="orange",ls="-",lw=1.5)
    # ax[1].plot(x,n3,color="black",ls="-",lw=1)
    #ax[1].fill_between(x,0,n2,color="none",hatch="////",edgecolor="wheat")
    ax[1].set_xlim([0,10])
    ax[1].set_xlabel("Annotated % of training dataset needed to reach BENCH",fontsize=fs-2,labelpad=9,va="center")
    ax[1].set_ylabel("Proportion of \n simulations",fontsize=fs-2,rotation=0,ha="center",labelpad=10)
    ax[1].yaxis.set_label_position("right")
    
    # plot_aucc(ax[2],n1,n2)
    
    # ax[2].set_xlabel("Query strategy",fontsize=fs,labelpad=-5)
    # ax[2].set_ylabel("AUCC",fontsize=fs,rotation=0,ha="center",labelpad=20)
    # ax[2].yaxis.set_label_position("right")
    # ax[2].set_title("(c) Area Under Cumulative Curve (AUCC)",fontsize=fs+1,loc="left")
    
    

    
    legend_elements = [Line2D([0], [0], color='steelblue', lw=4, label='Query strategy 1'),
                       Line2D([0], [0], color='orange', lw=4, label='Query strategy 2'),
                       Line2D([0], [0], lw=1, color='black',ls="--", label='Example curve',markerfacecolor='k', markersize=4),
                       Line2D([0], [0], marker='o',lw=0, color='black', label='BENCH crossing',markerfacecolor='black', markersize=4),
                       Line2D([0], [0], color='lightgray', lw=4, label="Range of curves"),
                       ]
    
    
    ax[0].legend(handles=legend_elements, loc='best',fontsize=fs-2,facecolor="white",edgecolor="none")
    
    iax = inset_axes(ax[1],width="20%",height="40%",loc="lower left",
                     bbox_to_anchor=(0.7,0,1,1), bbox_transform=ax[1].transAxes)


    iax.spines['top'].set_visible(False)
    iax.spines['right'].set_visible(False)
    iax.spines['bottom'].set_visible(True)
    iax.spines['bottom'].set_linewidth(0.5)
    iax.spines['left'].set_visible(False)
    iax.set_xticks([])
    iax.set_yticks([])
    iax.set_title("Area (AUCC)",fontsize=fs-2,bbox=dict(facecolor='white',edgecolor="none"))
    plot_aucc(iax,n1,n2)
    
    ax[1].axhline(y=0.775,xmin=0.215,color="k",lw=0.5)
    ax[1].axvline(x=2.1,ymax=0.75,color="k",lw=0.5)
    ax[1].axvline(x=5.25,ymax=0.75,color="k",lw=0.5)
    ax[1].set_xticks([0,2.1,5.25,10])
    ax[1].set_xticklabels(["0%","B","C","100%"],fontsize=fs-2)
    ax[1].set_yticks([0,0.775,1])
    ax[1].set_yticklabels(["0%","A","100%"],fontsize=fs-2)
    for a in ax.flatten():
        a.tick_params(width=0.5)
        
    fig.savefig(save_dir+"F_evaluation.pdf",
        dpi=300,
        bbox_inches='tight',
        format="pdf"
        )  