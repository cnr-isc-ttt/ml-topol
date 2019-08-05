import tensorflow as tf
import numpy as np
from sklearn import preprocessing
class Scaler:
    def __init__(self, d):
        self.scaler={}
        for key in d.keys():
            self.scaler[key] = preprocessing.StandardScaler()
            self.scaler[key].fit(d[key].reshape(-1,1))
    def __call__(self, x):
        X={}
        for key in x.keys():
            X[key]=self.scaler[key].transform(x[key].reshape(-1,1),copy=True).reshape(-1)
        return X
    def rescale(self, x): 
        X={}
        for key in x.keys():
            X[key]=self.scaler[key].inverse_transform(x[key].reshape(-1,1),copy=True).reshape(-1)
        return X
    def rescaleAs(self, key, x):
        return self.scaler[key].inverse_transform(x.reshape(-1,1),copy=True).reshape(-1)
    
class MyInputFn:
    def __init__(self, scaler, x, y=None):
        self.scaler=scaler
        self.x={}
        for key in x.keys():
            self.x[key]=np.copy(x[key])
        if y is not None:            
            self.y={}
            for key in y.keys():
                self.y[key]=np.copy(y[key])
        else:
            self.y=y
        
    def __call__(self, 
                 num_epochs=1,
                 shuffle=False,
                 batch_size=128,
                 queue_capacity=1000,
                 num_threads=1):
        x=self.scaler(self.x)
        if self.y is None:
            y=None
        else:
            y=self.scaler(self.y)
            y=y[y.keys().pop()] # from single y dict to array
        return tf.estimator.inputs.numpy_input_fn(x, y, batch_size, num_epochs,shuffle,queue_capacity,num_threads)
    
    def target(self):        
        y=self.y
        y=y[y.keys().pop()] # from y dict to array        
        return y
        