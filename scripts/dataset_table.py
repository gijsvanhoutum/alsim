import os
import sys
sys.path.append(os.path.split(os.getcwd())[0])

import numpy as np
import pandas as pd

from aml.dataset import load_dataset
import string

caption = """{} open-source datasets from the OpenML database and {} datasets 
from {} different directed-energy-deposition (DED) processes. "I" is the 
number of data instances in the dataset, "F" is the number of features per 
instance and "C" is the number of unique classes in the dataset."""

def to_dataframe(dataset,alias=False):
    
    data = []
    for nv,arr in dataset.items():
        
        name = nv.split("_")[0]
        n_features = arr.shape[1]-1
        n_samples = arr.shape[0]
        n_classes = len(np.unique(arr[:,-1]))
        row = (name,n_samples,n_features,n_classes)
        data.append(row)    
    
    df=pd.DataFrame(data=data,columns=["Dataset","I","F","C"])
    df = df.sort_values("I")
    
    if alias:
        n_data = len(dataset)
        alias = ["{DED-"+str(i)+"}" for i in string.ascii_lowercase[:n_data]]
        df["Dataset"] = alias
    
    
    return df
    
if __name__=="__main__":
    cwd = os.getcwd()
    
    data_dir = cwd+r'\data'
    
    save_dir = data_dir+r"\visualization"
    
    dual_column = True
    
    source_dir = data_dir+r"/compressed/"

    
    size = "scriptsize"
    colsep=3
    
    dx = load_dataset( source_dir, date=-1,hint="xiris")[0]
    do = load_dataset( source_dir, date=-1,hint="openml")[0]

    dfx = to_dataframe(dx,alias=True)
    dfo = to_dataframe(do)
    df = pd.concat([dfx,dfo],ignore_index=True)
    df.sort_values("I",inplace=True)

    n = len(df)
    
    if dual_column:
        df1,df2 = df[:n//2],df[n//2:]
        df = df1.reset_index(drop=True).merge(df2.reset_index(drop=True),
                                              left_index=True, right_index=True)
        
    caption = caption.format(len(dfo),len(dfx),len(dfx) )
    lt = df.to_latex(index=False,caption=caption)
    lt = lt.replace("\{","\\textbf{")
    lt = lt.replace("\}","}")
    
    lt = lt.replace("\_x","" )    
    lt = lt.replace("\_y","" ) 
    lt = lt.replace("\centering","\\"+size+"\n\centering\n\\begin{threeparttable}\n\setlength{\\tabcolsep}{"+str(colsep)+"pt}")
    lt = lt.replace("\\end{table}","\\end{threeparttable}\n\\end{table}" )

    with open(save_dir+"datasets.tex", "w") as text_file:
        text_file.write(lt)   