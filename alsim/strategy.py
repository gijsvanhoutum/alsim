import numpy as np
from numpy.linalg import norm
import random
import bisect
from scipy.spatial.distance import cdist
from abc import ABC, abstractmethod

class QueryStrategy(ABC):
    """
    Query strategy base class
    """
    def reset(self,state):
        pass
    
    @abstractmethod
    def act(self, state,number=1):
        pass
    
    # FASTER implementation to np.random.choice ( single select only )
    def single_choice(self,p):
        return bisect.bisect_right(np.cumsum(p/p.sum()),random.random())

class RND(QueryStrategy):
    """
    Uniform random sampling
    """
    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        
    def act(self,state,number=1):   
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        w = np.ones(n_unknown)
        
        indices = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            index = self.single_choice(w)
            w[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices] 

class US(QueryStrategy):
    """
    Uncertainty sampling.
    """
    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        
    def act(self,state,number=1):   
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        s = self.score(state.probas(state.X[unknown]))
        
        indices = np.zeros(number,dtype=int)
        
        for i in np.arange(number):  
            w = (s==s.max()).astype(int)
            index = self.single_choice(w)
            s[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices]  
        
    def score(self, probas):  
        return 1.0 - probas.max(axis=1)+1e-10

class WUS(US):
    """
    Proportionally weighted uncertainty sampling
    """
    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        
    def act(self,state,number=1): 
        unknown = ~state.known        
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        w = self.score(state.probas(state.X[unknown]))
        
        indices = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            index = self.single_choice(w)
            w[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices] 

class AWUS_R(US):
    """
    van Houtum - 2021 - Adaptive Weighted Uncertainty Sampling
    """
    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=0, high=shape[1], size=shape[0])
        self.epsilon = 0.001
        self.rng = np.arange(shape[0])
    
    def ratio_similarity(self,probas):
        H_n = probas.argmax(axis=1)
        d = np.count_nonzero(H_n==self.H_o) / len(H_n)
        e = 1.0 / np.maximum(1-d,self.epsilon) - 1.0    
        self.H_o = H_n
        return e        
    
    def act(self,state,number=1):   
        unknown = ~state.known      
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)    
        
        probas = state.probas(state.X)
        e = self.ratio_similarity( probas )
        s = self.score(probas[unknown]) 
        n = s * probas.shape[1] / (probas.shape[1]-1)
        w = (n+1.0)**e
        
        indices = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            index = self.single_choice(w)
            w[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices] 

class AWUS_C(US):
    """
    van Houtum - 2021 - Adaptive Weighted Uncertainty Sampling
    """
    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.epsilon = 0.001
        self.rng = np.arange(shape[0])

    def cosine_similarity(self,probas):
        H_n = probas.argmax(axis=1)+1
        d = np.dot(self.H_o,H_n) / (norm(self.H_o)*norm(H_n))
        e = 1.0 / np.maximum(1-d,self.epsilon) - 1.0        
        self.H_o = H_n
        return e    
    
    def act(self,state,number=1):   
        unknown = ~state.known      
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)    
        
        probas = state.probas(state.X)
        e = self.cosine_similarity( probas )
        s = self.score(probas[unknown]) 
        n = s * probas.shape[1] / (probas.shape[1]-1)
        w = (n+1.0)**e
        
        indices = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            index = self.single_choice(w)
            w[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices] 
    
class BEE(US):
    """
    Osugi - 2005 - Balancing Exploration and Exploitation: A New Algorithm 
    for Active Machine Learning
    """
    def __init__(self,l=0.1,e=0.01):
        self.l = l # lambda parameter
        self.e = e # error parameter

    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.p = 1.0
        self.rng = np.arange(shape[0])
    
    def act(self,state,number=1):   
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)

        probas = state.probas(state.X)
        H_n = probas.argmax(axis=1)+1
        s = np.dot(self.H_o,H_n) / (norm(self.H_o)*norm(H_n))
        d = 3-4*s
        self.p = max(min(self.p*self.l*np.exp(d),1.0-self.e),self.e)
        
        indices = np.zeros(number,dtype=int)

        s = self.score(probas[unknown])

        for i in np.arange(number):

            if random.random() < self.p:
                w = (s>0).astype(int)
            else:
                w = (s==s.max()).astype(int)

            index = self.single_choice(w)
            s[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices]              
            
    
class EGA(US):
    """
    Bouneffouf - 2016 - Exponentiated Gradient Exploration for
    Active Learning
    """
    def __init__(self,num=11):
        self.b = 1e-10
        self.t = 1e-10
        self.e = np.linspace(0,1,num=num)
        self.T = num
        self.w = np.ones(num)
        self.p = self.w / num

    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.d = np.random.randint(low=0, high=self.T, size=1)
        self.rng = np.arange(shape[0])
                
    def act(self,state,number=1):  
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        probas = state.probas(state.X)
        H_n = probas.argmax(axis=1)+1    
        dr = np.dot(self.H_o,H_n) / (norm(self.H_o)*norm(H_n))
        self.H_o = H_n
        r = 2*np.arccos(np.minimum(1,dr)) / np.pi

        for k in np.arange(self.T):
            I = (k==self.d).astype(int)
            self.w[k] *= np.exp(self.t*(r*I+self.b) / self.p[k])

        self.d = self.single_choice(self.w)
        s = self.score(probas[unknown])
        
        indices = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            
            if random.random() < self.e[self.d]:
                w = (s>0).astype(int)
            else:
                w = (s==s.max()).astype(int)

            index = self.single_choice(w)
            s[index] = 0
            indices[i] = index
            
        return self.rng[unknown][indices]              
            
            
class UDD(US): 
    """
    Kee - 2018 - Query-by-committee improvement with diversity 
    and density in batch active learning
    """
    def __init__(self,den=1.0/3,div=1.0/3,metric="cosine"):   
        self.m = metric
        self.w = np.array([1.0-den-div,div,den],dtype=float)
        
    def reset(self,state):
        self.d = cdist(state.X,state.X,metric=self.m)
        self.h = -self.d.sum(axis=1)
        self.h -= self.h.min()
        self.rng = np.arange(state.X.shape[0])
        
    def act(self,state,number=1): 
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        s = np.zeros((n_unknown,3),dtype=float)
        s[:,0] = self.score(state.probas(state.X[unknown]))
        s[:,1] = self.d[:,~unknown][unknown].min(axis=1)
        s[:,2] = self.h[unknown] 

        indices = np.zeros(number,dtype=int)
        bp = self.rng[unknown]
        for i in np.arange(number):
            mx = s.max(axis=0)
            nz = mx>0
            u = np.matmul(s[:,nz] / mx[nz] ,self.w[nz])
            w = (u==u.max()).astype(float)
            index = self.single_choice(w)     
            s[:,1] = np.minimum(s[:,1], self.d[:,bp[index]][unknown])
            s[index] = 0
            indices[i] = index
            
        return bp[indices]  

    