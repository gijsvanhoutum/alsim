import numpy as np
from numpy.linalg import norm
import random
from scipy.spatial.distance import cdist

from abc import ABC, abstractmethod


class QueryStrategy(ABC):
    """
    Query strategy base class
    """
    def __init__(self,name):
        self._name = name
        
    @property
    def name(self):
        return self._name
    
    def reset(self,state):
        pass
    
    @abstractmethod
    def act(self, state,number=1):
        pass
    
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
        indices = np.arange(n_unknown)
        np.random.shuffle(indices)       
        actions = indices[:number]
        actions = self.rng[unknown][actions]
        return actions
    
class US(QueryStrategy):
    """
    Uncertainty sampling.
    """
    def __init__(self,name,method="max"):
        super().__init__(name)
        self.map = {'max':self.max,
                    'entropy':self.entropy}
        
        self.score = self.map[method]

    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        
    def act(self,state,number=1):   
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        indices = np.arange(n_unknown)
        np.random.shuffle(indices)
        s,n = self.score(state.probas(state.X[unknown]))
        idx = np.argsort(s[indices])[-number:][::-1]
        actions = self.rng[unknown][indices][idx]
        return actions    
        
    def max(self, probas):  
        raw = 1.0 - probas.max(axis=1)
        norm = raw * probas.shape[1] / (probas.shape[1]-1)
        return raw,norm

    def entropy(self, probas):
        raw = -np.sum(probas*np.log(probas),axis=1)
        norm = raw / np.log(probas.shape[1])
        return raw,norm
    
class RUS(US):
    """
    Randomly choose between uniform random and uncertainty sampling
    """
    def __init__(self,name,p=0.5,method="max"):
        super().__init__(name,method=method)
        
        self.p = p # uniform random sampling probability

    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        self.min = 1e-10
        
    def act(self,state,number=1):   
        unknown = ~state.known    
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)

        s,n = self.score(state.probas(state.X[unknown]))
        s+=self.min
        actions = np.zeros(number,dtype=int)

        for i in np.arange(number):  
            if random.random() < self.p:
                w = (s>0).astype(int)
            else:
                w = (s==s.max()).astype(int)

            p = w / np.count_nonzero(w)
            index = np.random.multinomial(1,p).astype(np.bool)
            actions[i] = self.rng[unknown][index] 
            s[index] = 0
            
        return actions

class WUS(US):
    """
    Proportionally weighted uncertainty sampling
    """
    def reset(self,state):
        self.rng = np.arange(state.X.shape[0])
        self.min = 1e-10
        self.rg = np.random.default_rng()
        
    def act(self,state,number=1): 
        unknown = ~state.known        
        n_unknown = np.count_nonzero(unknown)
        nr = np.minimum(n_unknown, number)
        s,n = self.score(state.probas(state.X[unknown]))
        s+=self.min
        p = s/s.sum()
        actions = self.rg.choice(self.rng[unknown],size=nr,replace=False,p=p)
        return actions
    
class AWUS(US):
    """
    Model change weighted uncertainty sampling
    """
    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.epsilon = 0.001
        self.rng = np.arange(shape[0])
        self.rg = np.random.default_rng()

    def update(self,state,H_o): 
        probas = state.probas(state.X)
        H_n = probas.argmax(axis=1)+1
        s = np.dot(H_o,H_n) / (norm(H_o)*norm(H_n))
        d = np.arccos(np.minimum(1.0,s)) / np.pi
        e = 1.0 / np.maximum(d,self.epsilon) - 2.0
        return e,probas,H_n  
    
    def act(self,state,number=1): 
        e,probas,self.H_o = self.update(state,self.H_o)       
        unknown = ~state.known      
        n_unknown = np.count_nonzero(unknown)
        nr = np.minimum(n_unknown, number)        
        s,n = self.score(probas[unknown]) 
        s = (n+1.0)**e
        p = s/s.sum()     
        actions = self.rg.choice(self.rng[unknown],size=nr,replace=False,p=p)
        return actions

    
class BEE(US):
    """
    Osugi - 2005 - Balancing Exploration and Exploitation: A New Algorithm 
    for Active Machine Learning
    """
    def __init__(self,name,l=0.1,e=0.01,method="max"):
        super().__init__(name,method=method)

        self.l = l # lambda parameter
        self.e = e # error parameter

    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.p = 1.0
        self.rng = np.arange(shape[0])
        
    def update_p(self,state,H_o,p,l,e):
        probas = state.probas(state.X)
        H_n = probas.argmax(axis=1)+1
        s = np.dot(H_o,H_n) / (norm(H_o)*norm(H_n))
        d = 3-4*s
        pn = max(min(p*l*np.exp(d),1-e),e)
        return pn,H_n,probas

    def act(self,state,number=1):   
        self.p,self.H_o,probas = self.update_p(state,self.H_o,self.p,self.l,self.e)
        
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)

        actions = np.zeros(number,dtype=int)

        s,n = self.score(probas[unknown])

        for i in np.arange(number):

            if random.random() < self.p:
                w = (s>-1).astype(int)
            else:
                w = (s==s.max()).astype(int)

            p = w / np.count_nonzero(w)
            index = np.random.multinomial(1,p).astype(np.bool)
            actions[i] = self.rng[unknown][index] 
            s[index] = -1
            
        return actions 

    
class EGA(US):
    """
    Bouneffouf - 2016 - Exponentiated Gradient Exploration for
    Active Learning
    """
    def __init__(self,name,b=0.1,t=0.1,num=11,method="max"):
        super().__init__(name,method=method)
        
        self.b = b
        self.t = t
        self.e = np.linspace(0,1,num=num)
        self.T = num
        self.w = np.ones(num)
        self.p = self.w / num

    def reset(self,state):
        shape = state.X.shape
        self.H_o = np.random.randint(low=1, high=shape[1]+1, size=shape[0])
        self.d = np.random.choice(self.T,size=1,p=self.p)
        self.rng = np.arange(shape[0])
        
    def update_p(self,state,H_o):
        probas = state.probas(state.X)
        H_n = probas.argmax(axis=1)+1    
        dr = np.dot(H_o,H_n) / (norm(H_o)*norm(H_n))
        d = np.minimum(1,dr)
        r = 2*np.arccos(d) / np.pi

        for k in np.arange(1,self.T+1):
            I = (k==self.d).astype(int)
            self.w[k-1] = self.w[k-1] *  np.exp(self.t*(r*I+self.b) / self.p[k-1])

        self.p = self.w / self.w.sum()

        return H_n,probas
                
    def act(self,state,number=1):   
        self.H_o,probas = self.update_p(state,self.H_o)
        
        unknown = ~state.known
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)

        s,n = self.score(probas[unknown])

        actions = np.zeros(number,dtype=int)
        
        self.d = np.random.choice(self.T,size=1,p=self.p)
            
        for i in np.arange(number):

            if random.random() < self.e[self.d]:
                w = (s>-1).astype(int)
            else:
                w = (s==s.max()).astype(int)

            p = w / np.count_nonzero(w)
            index = np.random.multinomial(1,p).astype(np.bool)
            actions[i] = self.rng[unknown][index] 
            s[index] = -1

        return actions               
            
            
class UDD(US): 
    """
    Kee - 2018 - Query-by-committee improvement with diversity 
    and density in batch active learning
    """
    def __init__(self,name,b=1/3,l=1/3,method="max"):
        super().__init__(name,method=method)
        
        self.b = b
        self.l = l
        
    def reset(self,state):
        self.d = cdist(state.X,state.X,metric="cosine")
        np.fill_diagonal(self.d,np.nan)
        self.h = 1-np.nanmean(self.d,axis=1)
        self.rng = np.arange(state.X.shape[0])
        
    def norm(self,arr):
        if arr.min() == arr.max():
            return arr*0+0.5
        else:
            a = arr - arr.min()
            b = a/a.max()
            return b
    
    def act(self,state,number=1): 
        unknown = ~state.known       
        n_unknown = np.count_nonzero(unknown)
        number = np.minimum(n_unknown, number)
        
        s,n = self.score(state.probas(state.X)) 
        
        actions = np.zeros(number,dtype=int)
        
        unk = np.copy(unknown)
        for i in np.arange(number):
            d = self.norm( self.d[:,~unk][unk].min(axis=1) )
            h = self.norm( self.h[unk] )
            f = self.norm( s[unk] )
            u = (1-self.l-self.b)*f+self.b*h+self.l*d
            w = (u==u.max()).astype(int)
            p = w / np.count_nonzero(w)
            index = np.random.multinomial(1,p).astype(np.bool)
            actions[i] = self.rng[unk][index] 
            unk[actions[i]] = False
            
        return actions
    