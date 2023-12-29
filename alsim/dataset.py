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

def save_dataset(dataset, source_path, save_dir):

    if not issubclass(dataset,Dataset):
        raise TypeError("Should be of class Dataset")    
    
    datasets = dataset.get_datasets( source_path )

    file_name = dataset.__id__+"-"+time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join(save_dir,file_name)   
     
    with open(file_path, 'wb') as handle:
        pickle.dump(datasets, handle, protocol=3)
            
def load_dataset( save_dir, hint="openml", date=-1):
    if not os.path.exists(save_dir):
        raise(FileNotFoundError("{} does not exist".format(save_dir)))
    
    file_names = [f for f in os.listdir(save_dir) if f.startswith(hint)]

    if abs(date)>len(file_names):
        return None
    
    file_name = sorted(file_names)[date]
    file_path = os.path.join(save_dir,file_name)

    with open(file_path, 'rb') as handle:
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
    
class XirisDataset(Dataset):
    __id__="xiris"
 
    @classmethod
    def _compress(cls,frame, dim ):
        frame = cls.min_max_scale(frame)
        hgm = cls._histogram_of_gradient_magnitudes( frame, dim )
        hcm = cls._histogram_of_intensity_magnitudes( frame, dim )
        hist = np.hstack( (hgm,hcm))
        return cls.log_norm_transform(hist).astype(np.float32)
    
    @classmethod
    def log_norm_transform(cls,hist):      
        return np.log( hist+ 1 ) / np.log( hist.sum() / 2 + 1 ) -0.5
    
    @classmethod
    def min_max_scale(cls,frame):
        frame = frame.astype(np.float32)
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
        return hist.astype(np.uint64)

    @classmethod
    def _histogram_of_intensity_magnitudes(cls, frame, dim ):
        hist = cv2.calcHist([frame],[0],None,[dim],[0,1]).ravel()
        return hist.astype(np.uint64)
        
    @classmethod
    def get_datasets(cls, source_path, dim=8 ):
        
        datasets = []
        
        names = []
        for video_name in tqdm( os.listdir( source_path ) ):   
            video_path = os.path.join( source_path, video_name)
            
            X,I,y = [],[],[]
            for i,class_name in enumerate( os.listdir( video_path ) ):
                class_path = os.path.join( video_path, class_name)     
                for frame_name in os.listdir( class_path ):
                    frame_path = os.path.join( class_path, frame_name)
                    PIL_image = Image.open(frame_path)
                    np_image = np.array(PIL_image)
                    
                    I.append(np_image)
                    X.append( cls._compress( np_image, dim=dim) )
                    y.append( class_name )

            names.append(video_name)
            datasets.append( {"X":np.vstack(X).astype(np.float64),
                              "I":np.array(I).astype(np.uint8),
                              "y":np.array(y).astype(str)} )
            
        
        lengths = [len(d["y"]) for d in datasets]
        inds = np.argsort(lengths)
        strs = string.ascii_lowercase[:len(names)]
        datasets = {"DED-"+strs[i]:datasets[inds[i]] for i in range(len(inds))}
        return datasets