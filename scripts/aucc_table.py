import os
import pandas as pd
import numpy as np

def mm2inch(*tupl):
    inch = 25.4
    if len(tupl)>1:
        return tuple(i/inch for i in tupl)
    else:
        return tupl[0]/inch   
    
def hist(data, bins=10):
    H, bins = np.histogram(data, bins=bins)#, range=rng)
    bins = np.concatenate(([bins[0]],bins,[bins[-1]]))
    H = np.concatenate(([0],[H[0]],H,[0]))
    return H,bins

def aucc(df,step=0.01):
    r = np.arange(0,1+step,step)
    H,_ = np.histogram(df["percentage"], bins=r)
    H = np.cumsum(np.insert(H,0,0))
    H = H/H.max() 
    return np.trapz(H)/len(r) 

def rank(vals):
    rnk = np.argsort(vals)
    vals = np.round(vals*100,0).astype(int)
    out = vals.astype(str)
    out[rnk[-1]] = r"\bf{{{}}}".format(vals[rnk[-1]])
    return out

    
if __name__=="__main__":
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'
    
    save_dir = data_dir+r"\visualization"
    save_name = "xiris"
    N=4
    fs=6

    df = pd.read_pickle(data_dir+r"\simulation\results2.pkl")

    df["percentage"] = df["k"] / df["kmx"]

    sind = df.drop_duplicates(subset=["dataset","kmx"])["dataset"].values[::-1]

    df2 = df.groupby(["dataset","classifier","agent"]).apply(aucc).unstack("agent")


    aucc_ = df2.unstack("classifier")
    adds = ["Ave. OpenML","Ave. DED","Ave. Overall"]
    key = "DED"
   
    aucc_.loc[adds[0]] = aucc_.loc[~aucc_.index.str.startswith(key, na=False) ].mean(axis=0)
    aucc_.loc[adds[1]] = aucc_.loc[aucc_.index.str.startswith(key, na=False) ].mean(axis=0)
    aucc_.loc[adds[2]] = aucc_.mean(axis=0)
    
    aucc_ = aucc_.stack("classifier")
    aucc_ = aucc_.apply(rank,axis=1,raw=True)
    aucc_ = aucc_.unstack("classifier")
    aucc_ = aucc_.reindex(np.concatenate((sind, adds)))
    aucc_.columns = aucc_.columns.swaplevel(0, 1)
    aucc_.sort_index(axis=1, level=0, inplace=True)
    
    clfs = np.unique(df["classifier"])
    cf = "l"
    cm = "\cmidrule(r){2-8}\cmidrule(r){9-15}\cmidrule(r){16-22}\cmidrule(r){23-29}"
    for i in range(len(clfs)):
        cf += "@{\hskip 2\\tabcolsep}ccccccc"

    lt = aucc_.to_latex(caption="test",multicolumn_format="c",column_format=cf,index_names=False)
    lt = lt.replace("textbackslash ","")
    lt = lt.replace("\}","}")
    lt = lt.replace("\{","{")
    lt = lt.replace("\$","")
    lt = lt.replace("\centering","\\label{tab: aucc}\n\\scriptsize\n\\centering\n\setlength{\\tabcolsep}{1.25pt}") 
    lt = lt.replace("\\begin{table}","\\begin{table}" )
    lt = lt.replace("\\end{table}","\\end{table}" )
    
    lt = lt.replace("agent",cm)
    lt = lt.replace("classifier","")
    lt = lt.replace("Ave. OpenML","\\midrule\nAve. OpenML") 
    lt = lt.replace("\\textasciicircum ","^")  
    lt = lt.replace("\\\\\n{}","\\\\\n {} \ndataset".format(cm))  
    
    with open(save_dir+r"\aucc.tex", "w") as text_file:
        text_file.write(lt)  