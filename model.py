import numpy as np
import tensorflow as tf

def createModelFromInput_fn(train_input_fn, hiddenLayers, maxSteps):
    featureColumns=tf.contrib.learn.infer_real_valued_columns_from_input_fn(train_input_fn)
    optimizer=tf.train.AdamOptimizer()
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=featureColumns, hidden_units=hiddenLayers,optimizer=optimizer,activation_fn=tf.nn.tanh)
    regressor.fit(input_fn=train_input_fn, max_steps=maxSteps) 
    return regressor

def Score(regressor, input_fn):
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
    return regressor.evaluate(input_fn=input_fn, steps=1, 
                           metrics={
                               'MSE': tf.contrib.metrics.streaming_mean_squared_error
                           })

def SelfConsistantCycle(directRegressor, inverseRegressor, scaler, omegaSweep_input_fn_Inverse):    
    from myInputFn import MyInputFn
    from myInputFn import Scaler
    omegaSweep=np.copy(omegaSweep_input_fn_Inverse.x['omega'])
    gammaSweep=np.copy(omegaSweep_input_fn_Inverse.x['gamma'])
    trendSweep=np.copy(omegaSweep_input_fn_Inverse.x['trend'])
    # https://github.com/tensorflow/tensorflow/issues/9505
    chiPredicted=scaler.rescaleAs('chi',np.asarray(list(inverseRegressor.predict_scores(input_fn=omegaSweep_input_fn_Inverse(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale
    predict_input_fn_Direct = MyInputFn( scaler,
        {'chi': chiPredicted, 'gamma': gammaSweep, 'trend': trendSweep})        
    omegaPredicted=scaler.rescaleAs('omega',np.asarray(list(directRegressor.predict_scores(input_fn=predict_input_fn_Direct(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale
    return chiPredicted,omegaPredicted
    
    
    
