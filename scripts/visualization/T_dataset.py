import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])

import numpy as np
import pandas as pd

from aml.dataset import load_dataset
import string

caption = """{} open-source datasets from the OpenML database and {} datasets 
from {} different directed-energy-deposition (DED) processes. The dataset name 
(N), number of instances (I), features (F) and unique classes (C) per dataset 
are presented on the left. F1-macro classification performance, mean and 
standard deviation in \%, on the validation set ( 50\%-50\% train-validation split) 
on the right over {} repeated random dataset splits"""

def to_dataframe(dataset):
    
    dataframe = []
    for name,data in dataset.items():
        n_features = data["X"].shape[1]
        n_samples = data["X"].shape[0]
        n_classes = len(np.unique(data["y"]))
        row = (name,n_samples,n_features,n_classes)
        dataframe.append(row)    
    
    df=pd.DataFrame(data=dataframe,columns=["N","I","F","C"])
    df = df.sort_values("I")
    
    return df

def bold(df):
    vals = df[-4:].values
    means = [float(v.split("$")[0]) for v in vals]
    am = np.argmax(means)
    vals[am] = r'\bf{'+str(vals[am])+"}"
    df[-4:] = vals
    if df[0].startswith("DED"):
        df[0] = r"\underline{"+str(df[0])+"}"
    return df
    
    
    
    
if __name__=="__main__":
    
    dual_column = True
    N=4
    fs=5

    data_dir = "../data"
    save_dir = data_dir+"/visualization"
    comp_dir = data_dir+"/compressed"
    data_path = data_dir+"/simulation/results.pkl"
    save_name = "xiris"
    N=4
    fs=6

    df = pd.read_pickle(data_path)

    df["percentage"] = df["k"] / df["kmx"]
    
    
    n_clf = len(df["classifier"].unique())
                
    ticks = np.arange(0,1.1,0.1).round(decimals=1)
    labels = sorted(df["agent"].unique())
    
    lookup_2 = {"rf":"RF",
                "gnb":"GNB",
                "lr":"LR",
                "svm":"SVM"}
    
    size = "scriptsize"
    colsep=3
    
    dss1 = load_dataset( comp_dir, date=-1, hint="openml")
    dss2 = load_dataset( comp_dir, date=-1, hint="xiris")
    
    dfx = to_dataframe(dss2)
    dfo = to_dataframe(dss1)
    dataset = pd.concat([dfx,dfo],ignore_index=True)
    dataset.sort_values("I",inplace=True)

    df = df.groupby("agent").get_group("rnd")
    df = df.groupby("batch").get_group(1)
    
    for clf,dfc in df.groupby("classifier"):
        dsg = dfc.groupby("dataset")
        Q=[]
        for name,vals in dataset.groupby("N"):
            
            dfd = dsg.get_group(name)
            mean = np.round(dfd["qmx"].mean() *100.0,1)
            std = np.round(dfd["qmx"].std() *100.0,1)
            Q.append(r'{}$\pm${}'.format(mean,std))

        dataset[clf] = Q
        
    dataset = dataset.apply(bold,axis=1)

    cf = "llll"
    cm = "\cmidrule(r){1-4}\cmidrule(l){5-"+str(n_clf+4)+"}"
    for i in range(n_clf):
        cf += "c"
        
    colsep=5
    caption = caption.format(len(dfo),len(dfx),len(dfx),len(dfd))
    lt = dataset.to_latex(index=False,caption=caption,column_format=cf,multicolumn=False)
    lt = lt.replace("textbackslash ","")
    lt = lt.replace("\}","}")
    lt = lt.replace("\{","{")
    lt = lt.replace("\$","$")
    lt = lt.replace("\centering","\\label{tab: datasets}\n\\scriptsize\n\\centering\n\setlength{\\tabcolsep}{"+str(colsep)+"pt}") 
    lt = lt.replace("N ",cm+"\n N ")
    lt = lt.replace("toprule","toprule \n"+r"\multicolumn{4}{c}{Dataset}  & \multicolumn{4}{c}{Classifier} \\")

    with open(save_dir+"/T_dataset.tex", "w") as text_file:
        text_file.write(lt)   