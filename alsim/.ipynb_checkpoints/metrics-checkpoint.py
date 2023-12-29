import numpy as np

def f1_macro(y_true,y_pred,labels):

    a = y_pred == labels
    b = y_true == labels
    tp = np.count_nonzero(a & b,axis=1).astype(np.float32)
    fp = np.count_nonzero(a & ~b,axis=1).astype(np.float32)
    fn = np.count_nonzero(~a & b,axis=1).astype(np.float32)
    
    tpfp = tp+fp
    tpfp[tpfp==0] = np.nan

    tpfn = tp+fn
    tpfn[tpfn==0] = np.nan
    
    p = tp / tpfp
    r = tp / tpfn

    pr = p+r
    pr[pr==0] = np.nan
    
    f1s = 2*p*r / pr
    mac_f1 = np.nanmean(f1s)
    return mac_f1

def accuracy_score(y_true,y_pred,labels):
    return np.count_nonzero(y_true==y_pred) / np.size(y_true)
