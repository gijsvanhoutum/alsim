import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from functools import partial

class State:

    def __init__(self,X,model,known,done):

        self._X = X      
        self._model = model
        self._known = known
        self._done = done
       
    @property
    def X(self):
        return self._X

    @property
    def done(self):
        return self._done
    
    @property
    def known(self):
        return self._known
    
    def probas(self,X):
        return self._model.predict_proba(X)
    
    def predict(self,X):
        return self._model.predict(X)
    
class Environment: 

    def __init__(self,model,data,quality,ratio=0.5,batch=True,stop=None):
    
        _, self.y = np.unique(data["y"], return_inverse=True)
        self.X = data["X"].astype(np.float32)
        self.uy = np.unique(self.y)
        self.model = model()
        self.batch = batch
        self.ratio = ratio
        self.stop = stop
        self.quality = partial(quality,labels=self.uy[:,None])
    
    def reset(self):
        train,test = self._train()       
        self._split(train,test)  
        self.known = np.ones(len(self.y_trn),dtype=np.bool)
        self.max_q = self.get_quality()
        self.train_nr = len(self.y_trn)
        self.known = self._initial(train)
        self.episode_qualities = {"known":[],"quality":[]}       
        self.cur_max_q = 0.0
        self.done = self._update()      
        return self._state()

    def _split(self, train,test ):    
        scaler = StandardScaler() 
        scaler = MinMaxScaler(feature_range=(-1,1))
        self.X_trn = scaler.fit_transform(self.X[train])
        self.y_trn = self.y[train]
        self.X_tst = scaler.transform(self.X[test])
        self.y_tst = self.y[test]
    
    def _sample(self,mask,nr):
        p = mask / mask.sum()
        ids = np.random.choice(len(mask),size=nr,replace=False,p=p)
        return ids
                    
    def _train(self): 
        if self.ratio <= 0 or self.ratio >= 1:
            train = test = np.ones(self.y.size,dtype=np.bool)
        else:
            train = np.zeros(self.y.size,dtype=np.bool)
            for k in self.uy:
                m = (self.y==k).astype(int)
                nr = int(np.count_nonzero(m) * self.ratio)
                ids = self._sample(m,nr)
                train[ids] = True    
                
            test = ~train
                        
        return train,test

    def _initial(self, train ):
        known = np.zeros(self.y.size,dtype=np.bool) 
        for k in self.uy:
            m = np.logical_and(self.y==k,train).astype(int)
            idx = self._sample(m,1)
            known[idx] = True
                       
        return known[train]
    
    def get_quality(self):
        self.model.fit(self.X_trn[self.known],self.y_trn[self.known])
        y_prd = self.model.predict(self.X_tst)  
        qlt = self.quality(self.y_tst,y_prd)        
        return float(qlt)
        
    def _update(self):
        qlt = self.get_quality()
        self.cur_max_q = float(np.maximum(self.cur_max_q,qlt))
        self.n_known = int( np.count_nonzero(self.known)  )  
        self.episode_qualities["quality"].append(qlt)
        self.episode_qualities["known"].append(self.n_known) 
        
        if self.stop <=1 and self.cur_max_q >= self.max_q*self.stop:
            self.episode_qualities["quality"].append(self.max_q)
            self.episode_qualities["known"].append(self.train_nr) 
            return True
        elif self.stop >= self.n_known or self.train_nr == self.n_known:
            return True
        else:
            return False

    def _state(self):
        return State(self.X_trn,self.model,self.known, self.done)
    
    def qualities(self):
        return self.episode_qualities
        
    def _check(self, actions ):
        if not isinstance( actions ,np.ndarray):
            raise TypeError("Wrong index type: {} should be [list]".format(type(actions)))          

        if len(set(actions)) != len(actions):
            raise TypeError("Actions should be unique")  
            
        if np.count_nonzero(self.known[actions]):
            raise ValueError("Indices already known")
        
        if self.n_known >= self.train_nr:
            raise ValueError("No actions left!")
            
        if any(i < 0 for i in actions):
            raise ValueError("Index negative")

        if any(i > self.train_nr-1 for i in actions):
            raise ValueError("Index too high")
        
    def step(self, actions):  
        self._check( actions )
        
        if self.batch:        
            self.known[actions] = True 
            self.done = self._update()
        else:
            for a in actions:
                self.known[a] = True 
                self.done = self._update()
                if self.done: break
            
        return self._state() 