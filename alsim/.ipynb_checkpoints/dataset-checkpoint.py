import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from abc import ABC, abstractmethod
from sklearn.datasets import fetch_openml
import pickle
import string

def save_load_dataset(dataset, source_path, save_path):

    if not issubclass(dataset,Dataset):
        raise TypeError("Should be of class Dataset")    
    
    if not os.path.exists(save_path):
        
        datasets = dataset.get_datasets( source_path )
        
        with open(save_path, 'wb') as handle:
            pickle.dump(datasets, handle, protocol=3)
    else:
        with open(save_path, 'rb') as handle:
            datasets = pickle.load(handle)
            
    return datasets
        
class Dataset(ABC):

    @property
    @classmethod
    @abstractmethod
    def __id__(cls):
        return NotImplementedError
        
    @classmethod
    @abstractmethod
    def get_datasets(self):
        pass

class OpenmlDataset(Dataset):
    __id__ = "openml"
    
    @classmethod
    def _get_dict(cls, openml_src ):
        with open(openml_src) as f:
            names_list = f.readlines()
        
        name_dict = {}
        for name_str in names_list:
            r  = name_str.replace("\n",",")
            s = r.split(",")
            name_dict[ s[0] ] = int(s[1])
            
        return name_dict
    
    @classmethod
    def get_datasets(cls, file_path ):
        name_dict = cls._get_dict( file_path )
            
        datasets = {}
        for n,v in tqdm( name_dict.items() ):
            X,y = fetch_openml(name=n,version=v,return_X_y=True,as_frame=False)     
            _,y = np.unique(y, return_inverse=True)
            datasets[n] = {"X":X.astype(np.float64),
                           "y":y.astype(np.uint8)}
        return datasets