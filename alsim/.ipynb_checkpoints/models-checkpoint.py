import numpy as np
import scipy as sp
from scipy.special import logsumexp
from sklearn.svm import _liblinear as liblinear
from sklearn.tree._tree import DOUBLE
import numbers

from warnings import warn
from sklearn.tree._classes import CRITERIA_CLF,DENSE_SPLITTERS
from sklearn.tree._tree import DepthFirstTreeBuilder,BestFirstTreeBuilder,Tree,_build_pruned_tree_ccp
from math import ceil

class FastGNB:

    def __init__(self, *, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return np.exp( jll - np.atleast_2d(log_prob_x).T )

    def _mean_var(self,n_past, mu, var, X):

        if X.shape[0] == 0:
            return mu, var

        n_new = X.shape[0]
        new_var = np.var(X, axis=0)
        new_mu = np.mean(X, axis=0)

        if n_past == 0:
            return new_mu, new_var

        n_total = float(n_past + n_new)

        # Combine mean of old and new data, taking into consideration
        # (weighted) number of observations
        total_mu = (n_new * new_mu + n_past * mu) / n_total

        # Combine variance of old and new data, taking into consideration
        # (weighted) number of observations. This is achieved by combining
        # the sum-of-squared-differences (ssd)
        old_ssd = n_past * var
        new_ssd = n_new * new_var
        total_ssd = (old_ssd + new_ssd +
                     (n_new * n_past / n_total) * (mu - new_mu) ** 2)
        total_var = total_ssd / n_total

        return total_mu, total_var

    def fit(self, X, y):


        self.classes_ = np.unique(y)

        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))

        self.class_count_ = np.zeros(n_classes, dtype=np.float64)

        self.class_prior_ = np.zeros(len(self.classes_),dtype=np.float64)

        for y_i in self.classes_:
            i = self.classes_.searchsorted(y_i)
            X_i = X[y == y_i, :]

            N_i = X_i.shape[0]

            self.theta_[i, :],self.sigma_[i, :] = self._mean_var(
                                                        self.class_count_[i], 
                                                        self.theta_[i, :], 
                                                        self.sigma_[i, :],
                                                        X_i
                                                        )

            self.class_count_[i] += N_i

        self.sigma_[:, :] = self.sigma_[:,:]+self.var_smoothing

        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        joint_log_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])

            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) /(self.sigma_[i, :]), 1)

            joint_log_likelihood.append(jointi + n_ij)

        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood

class FastLR:

    def __init__(self, dual=False, tol=1e-4, C=1.0,solver_type=0,
                 intercept_scaling=1, max_iter=10000,verbose=0):

        # Use solver_type = 0 (logistic regression), or 2, (squard hinge SVM)
        self.dual = dual
        self.tol = tol
        self.C = C
        self.bias = intercept_scaling
        self.max_iter = max_iter
        self.verbose = liblinear.set_verbosity_wrap(verbose)
        self.solver_type = solver_type
        self.epsilon = 0.1
        self.issparse = False
        self.random_state = np.random.mtrand._rand.randint(np.iinfo('i').max)

    def decision_function(self, X):
        scores = np.dot(X,self.coef_.T) + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores
    
    def predict(self, X):
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        sp.special.expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
        
    def fit(self,X, y):
    
        classes, y_ind = np.unique(y, return_inverse=True)
        length = len(classes)
        if length < 2:
            raise ValueError("Solver needs samples of at least 2 classes")
            
        # LibLinear wants targets as doubles, even for classification
        # ensure writable array
        y_ind = np.require(y_ind.astype(np.float64), requirements="W")
        
        class_weight_ = np.ones(length,dtype=np.float64,order='C')
        sample_weight = np.ones(X.shape[0], dtype=np.float64, order='C')      
       
        raw_coef_, n_iter_ = liblinear.train_wrap(X, y_ind, self.issparse, 
                                self.solver_type, self.tol, self.bias, 
                                self.C,class_weight_, self.max_iter,
                                self.random_state,self.epsilon, sample_weight)
    
        n_iter_ = max(n_iter_)
        if n_iter_ >= self.max_iter:
            raise ValueError("Failed to converge, increase iterations.")
    
        self.coef_ = raw_coef_[:, :-1]
        self.intercept_ = self.bias * raw_coef_[:, -1]
        self.n_iter = np.array([n_iter_])
        self.classes_ = classes
        return self

class FastDT:

    def __init__(self, criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features="auto",
                 max_leaf_nodes=None,
                 random_state=np.random.mtrand._rand,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        
        self.initialize()
        
    def initialize(self):
        
        self.max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)
        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
            
        self.max_leaf_nodes = (-1 if self.max_leaf_nodes is None
                          else self.max_leaf_nodes)   

        if not isinstance(self.max_leaf_nodes,int):
            raise ValueError("max_leaf_nodes must be integral number but was "
                             "%r" % self.max_leaf_nodes)
        if -1 < self.max_leaf_nodes < 2:
            raise ValueError(("max_leaf_nodes {0} must be either None "
                              "or larger than 1").format(self.max_leaf_nodes))
            
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")

        if isinstance(self.min_samples_leaf, int):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            self.min_samples_leaf = self.min_samples_leaf
        else:  # float
            raise ValueError("min_samples_leaf must be at least int>=1")

        if isinstance(self.min_samples_split,int):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split*self.n_samples))
            min_samples_split = max(2, min_samples_split)

        self.min_samples_split = max(min_samples_split, 2*self.min_samples_leaf)

        if self.min_impurity_split is not None:
            warn("The min_impurity_split parameter is deprecated. "
                          "Its default value has changed from 1e-7 to 0 in "
                          "version 0.23, and it will be removed in 0.25. "
                          "Use the min_impurity_decrease parameter instead.",
                          FutureWarning)

            if self.min_impurity_split < 0.:
                raise ValueError("min_impurity_split must be greater than "
                                 "or equal to 0")
        else:
            self.min_impurity_split = 0
            
        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be greater than "
                             "or equal to 0")
     
    def get_max_features(self,n_features_):
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = max(1, int(np.sqrt(n_features_)))
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = n_features_
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * n_features_))
            else:
                max_features = 0

        if not (0 < max_features <= n_features_):
            raise ValueError("max_features must be in (0, n_features]")
            
        return max_features
    
    def get_min_weight_leaf(self,sample_weight, n_samples,X):
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf*n_samples)
        elif isinstance(sample_weight, numbers.Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=DOUBLE)
            min_weight_leaf = (self.min_weight_fraction_leaf*np.sum(sample_weight))
        else:
            raise ValueError("sample weight is not a number")
            
        return min_weight_leaf
    
    def fit(self, X, y,classes,n_classes,n_features,n_samples,n_outputs,
            sample_weight=None, X_idx_sorted=None):

        max_features = self.get_max_features(n_features)
        min_weight_leaf = self.get_min_weight_leaf( sample_weight,n_samples,X)

        # Build tree
        criterion = CRITERIA_CLF[self.criterion](n_outputs,n_classes)

        splitter = DENSE_SPLITTERS[self.splitter](criterion,
                                            max_features,
                                            self.min_samples_leaf,
                                            min_weight_leaf,
                                            self.random_state)

        # Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
        if self.max_leaf_nodes < 0:
            builder = DepthFirstTreeBuilder(splitter, 
                                            self.min_samples_split,
                                            self.min_samples_leaf,
                                            min_weight_leaf,
                                            self.max_depth,
                                            self.min_impurity_decrease,
                                            self.min_impurity_split)
        else:
            builder = BestFirstTreeBuilder(splitter, 
                                           self.min_samples_split,
                                           self.min_samples_leaf,
                                           min_weight_leaf,
                                           self.max_depth,
                                           self.max_leaf_nodes,
                                           self.min_impurity_decrease,
                                           self.min_impurity_split)

        self.tree_ = Tree(n_features,n_classes,n_outputs)
        builder.build(self.tree_, X, y, sample_weight)#, X_idx_sorted)
        
        # Prune the tree. 
        # https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        # IS TIME CONSUMING
        if self.ccp_alpha > 0.0:
            pruned_tree = Tree(n_features,n_classes,n_outputs)
            _build_pruned_tree_ccp(pruned_tree,self.tree_,self.ccp_alpha)
            self.tree_ = pruned_tree
        return self
    
    def predict_proba(self,X):
        proba = self.tree_.predict(X) 
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer
        return proba


class FastRF:

    def __init__(self,tree=FastDT,n_estimators=100,*args,**kwargs):

        self.trees = [tree(*args,**kwargs) for i in range(n_estimators)]
        if tree==FastDT:
            self.fit_func = self.fast_fit
        else:
            self.fit_func = self.ori_fit
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_.take(np.argmax(proba, axis=1), axis=0)

    def predict_proba(self, X):
        X32 = X.astype(np.float32)
        # avoid storing the output of every estimator by summing them here
        all_proba = np.zeros((X32.shape[0],self.n_classes_),dtype=np.float64)
        for t in self.trees:
            all_proba += t.predict_proba(X32)
            
        return all_proba / len(self.trees)
    
    def fit(self,X,y):
        if len(y) != X.shape[0]:
            raise ValueError("Number in y does not match number X")       
        
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.fit_func(X,y)
        
    def fast_fit(self,X,y):
        X32 = X.astype(np.float32)
        n_samples, n_features = X.shape
        n_classes = np.array([self.n_classes_], dtype=np.intp)
        y64 = y[:,None].astype(np.float64)
        y64 = np.ascontiguousarray(y64, dtype=DOUBLE) 
        for t in self.trees: 
            t.fit(X32,y64,self.classes_,n_classes,n_features,n_samples,1)

        return self
        
    def ori_fit(self, X, y):
        for t in self.trees: 
            t.fit(X,y,check_input=False)
            
        return self

class FastNB:

    def fit(self, X, y,smooth=1e-9):
        """Fit Gaussian Naive Bayes according to X, y"""

        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.n_classes = len(self.classes_)    
        
        self.theta_ = np.zeros((self.n_classes, n_features))
        self.sigma_ = np.zeros((self.n_classes, n_features))
        self.class_count_ = np.zeros(self.n_classes, dtype=np.float64)
        self.class_prior_ = np.zeros(self.n_classes,dtype=np.float64)

        for i in range( self.n_classes):

            X_i = X[y == self.classes_[i], :]

            self.theta_[i, :] = np.mean(X_i, axis=0)
            self.sigma_[i, :] = np.var(X_i, axis=0)
            self.class_count_[i] += X_i.shape[0]

        self.sigma_[:, :] += smooth * np.var(X, axis=0).max()
        # Empirical prior, with sample_weight taken into account
        self.class_prior_ = self.class_count_ / self.class_count_.sum()

        return self

    def _joint_log_likelihood(self, X):
        
        joint_log_likelihood = []
        
        for i in range(self.n_classes):
            
            jointi = np.log(self.class_prior_[i])
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma_[i, :]))
            
            n_ij -= 0.5 * np.sum(((X - self.theta_[i, :]) ** 2) / (self.sigma_[i, :]), 1)
            
            joint_log_likelihood.append(jointi + n_ij)
 
        return np.array(joint_log_likelihood).T

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        log_prob_x = logsumexp(jll, axis=1)
        return np.exp(jll - np.atleast_2d(log_prob_x).T )