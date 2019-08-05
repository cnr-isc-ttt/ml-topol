import numpy as np
import tensorflow as tf
import time
import sys
tooSmall=10
class Model:
    
    def __init__(self, dataset, hiddenLayers, maxSteps,model_dir, optimizer, debug):
        self.ok=False
        if len(dataset['chi'])<tooSmall: # too few points for ml watch out as there will be 10 point just for gamma
            return        
        self.debug=debug
        from myInputFn import Scaler
        # scaler
        self.scaler=Scaler(dataset) 
        # split
        from sklearn import model_selection
        featKeys=[]
        featTest=[]
        featTrain=[]
        if debug:
            rnd=4852
        else:
            rnd=np.random.randint((2**31)*2)
        
        for key, val in dataset.iteritems():
            train,test= model_selection.train_test_split(val, test_size=0.2 , random_state=rnd)
            if key=='chi' or key=='omega':
                if key=='omega':
                    omegaTrain=train
                    omegaTest=test
                else:
                    chiTrain=train
                    chiTest=test
            else:                
                featKeys.append(key)
                featTest.append(test)
                featTrain.append(train)
    
        from myInputFn import MyInputFn
        
        
        # Direct regressor
        directTrainFeatures={'chi': chiTrain}
        for i,key in enumerate(featKeys):
            directTrainFeatures[key]=featTrain[i]
            
        train_input_fn_Direct = MyInputFn( self.scaler,
            directTrainFeatures,
            {'omega':omegaTrain})
        
        startTime = time.time()    
        self.directRegressor = self.createModelFromInput_fn(train_input_fn_Direct(num_epochs=None,shuffle=True), hiddenLayers, maxSteps,model_dir+"/direct", optimizer)
        print('# Direct Fitting time: %f' % (time.time() - startTime))

        # test direct regressor
        directTestFeatures={'chi': chiTest}
        for i,key in enumerate(featKeys):
            directTestFeatures[key]=featTest[i]
        
        test_input_fn_Direct = MyInputFn( self.scaler,
            directTestFeatures,
            {'omega':omegaTest})

        from model import Score        
        self.scoreDirect=self.Score(self.directRegressor,test_input_fn_Direct(num_epochs=1,shuffle=False))
        print'# Direct MSE: '+str(self.scoreDirect['MSE'])

        # Inverse regressor
        invTrainFeatures={'omega': omegaTrain}
        for i,key in enumerate(featKeys):
            invTrainFeatures[key]=featTrain[i]
        
        train_input_fn_Inverse = MyInputFn( self.scaler,
            invTrainFeatures,
            {'chi':chiTrain})
        startTime = time.time()    
        self.inverseRegressor = self.createModelFromInput_fn(train_input_fn_Inverse(num_epochs=None,shuffle=True), hiddenLayers, maxSteps,model_dir+"/inverse", optimizer)
        print('# Inverse Fitting time: %f' % (time.time() - startTime))

        invTestFeatures={'omega': omegaTest}
        for i,key in enumerate(featKeys):
            invTestFeatures[key]=featTest[i]
        
        test_input_fn_Inverse = MyInputFn( self.scaler,
            invTestFeatures,
            {'chi':chiTest})        
        self.scoreInverse=self.Score(self.inverseRegressor,test_input_fn_Inverse(num_epochs=1,shuffle=False))
        print'# Inverse MSE: '+str(self.scoreInverse['MSE'])    
        
        # calc omega range BUG BUG 1 d lin
        if dataset.has_key('gamma'):
            self.OmegaMinGamma(dataset['omega'], dataset['gamma'])
        self.isChiPositive=np.min(dataset['chi'])>0.0
        self.ok=True

    def createModelFromInput_fn(self, train_input_fn, hiddenLayers, maxSteps, model_dir, optimizer):
        if self.debug:
            rnd=87897978
            cores=1
        else:
            rnd=None
            cores=0            
        my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_steps=maxSteps/10,
            keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
            tf_random_seed=rnd,
            # num_cores= cores
        )
                
        if(optimizer=='grad'):
            opt=tf.train.GradientDescentOptimizer(0.001)
        elif(optimizer=='prox'):
            opt=tf.train.ProximalGradientDescentOptimizer(0.001)
        else:
            opt=tf.train.AdamOptimizer()
        featureColumns=tf.contrib.learn.infer_real_valued_columns_from_input_fn(train_input_fn)
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=featureColumns, hidden_units=hiddenLayers,optimizer=opt,activation_fn=tf.nn.tanh, model_dir=model_dir,config=my_checkpointing_config)
        regressor.fit(input_fn=train_input_fn, max_steps=maxSteps) 
        return regressor
    
    def Score(self, regressor, input_fn):
        # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/MetricSpec
        return regressor.evaluate(input_fn=input_fn, steps=1, 
                               metrics={
                                   'MSE': tf.contrib.metrics.streaming_mean_squared_error
                               })
    def OmegaMinGamma(self, omega, gamma):
        u,v=np.unique(gamma, True)
        gammaVal=[]
        omegaMin=[]
        omegaMax=[]
        s=None
        for x in v: # select each gamma group
            if s is not None: # find offsets of gamma groups
                gammaVal.append(gamma[s])
                omegaMin.append(np.min(omega[s:x]))
                omegaMax.append(np.max(omega[s:x]))
            s=x
        gammaVal.append(gamma[s])
        omegaMin.append(np.min(omega[s:]))
        omegaMax.append(np.max(omega[s:]))
        # regression
        if len(gammaVal)<2:
            self.slopeOmn, self.interceptOmn = 0.0,np.min(omega)
            self.slopeOmx, self.interceptOmx=0.0,np.max(omega)
        else:
            from scipy import stats
            self.slopeOmn, self.interceptOmn, r_value, p_value, std_err =stats.linregress(gammaVal,omegaMin)
            self.slopeOmx, self.interceptOmx, r_value, p_value, std_err =stats.linregress(gammaVal,omegaMax)
        return {'gamma':np.array(gammaVal).reshape(-1), 'omegaMin':np.array(omegaMin).reshape(-1), 'omegaMax':np.array(omegaMax).reshape(-1)}
    
    def omegaRange(self, gamma):
        omn=self.slopeOmn*gamma+ self.interceptOmn
        omx=self.slopeOmx*gamma+ self.interceptOmx
        return omn, omx
    
    def SelfConsistantCycle(self, omegaSweep_input_fn_Inverse):    
        from myInputFn import MyInputFn
        from myInputFn import Scaler
        import math
        chiPredicted=self.scaler.rescaleAs('chi',np.asarray(list(self.inverseRegressor.predict_scores(input_fn=omegaSweep_input_fn_Inverse(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale
        chiUpperBoundIdx=np.abs(chiPredicted)<math.pi # keep only valid chi and correct 
        if self.isChiPositive:
            chiLowerBoundIdx=chiPredicted>0.0 
        else:
            chiLowerBoundIdx=chiPredicted<0.0 
        validChiIdx=np.logical_and(chiLowerBoundIdx,chiUpperBoundIdx)   
        chiPredicted=chiPredicted[validChiIdx]
        if(np.count_nonzero(chiPredicted)<2): # too few points
            return None,None, None 
        features={}

        for key in omegaSweep_input_fn_Inverse.x.keys():
            if key != 'omega':
                features[key]=np.copy(omegaSweep_input_fn_Inverse.x[key])[validChiIdx]
            else:
                deviation=np.copy(omegaSweep_input_fn_Inverse.x[key])[validChiIdx] # original omegas
        # should it return here?        
        features['chi']=chiPredicted
        predict_input_fn_Direct = MyInputFn( self.scaler, features)        
        omegaPredicted=self.scaler.rescaleAs('omega',np.asarray(list(self.directRegressor.predict_scores(input_fn=predict_input_fn_Direct(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale
        deviation=omegaPredicted[:]-deviation[:]
        return chiPredicted,omegaPredicted, deviation

def padding(x):
        mid=len(x)/2
        X=np.append(-x[mid:0:-1]+2*x[0],x) 
        X=np.append(X,-x[-2:mid:-1]+2*x[-1])
        return X

def extendSymetrically(dataset,targetKey):
    for i,key in enumerate(dataset):
        if(key!=targetKey):
            mid=len(dataset[key])/2
            # rightmost half
            # y0=dataset[targetKey][-1]
            extend=dataset[key][:mid-1:-1]            
            x0=extend[0]
            extend=x0-(extend-x0)
            extendY=dataset[target][:mid-1:-1]
            y0=extendY[0]
            extendY=y0-(extendY-y0)
            # BUG BUG multi D ??
            
    return
def test():
    def omega(x):
        return 1.0+x*x+x*x*x
    def dOmega(x):
        return x*(2+3*x)
    def trend(x):
        return np.sign(dOmega(x))
    def es():
        chi=np.linspace(-0.5,1.5,20, dtype=np.float32)
        ds={'chi':chi, 'omega':omega(chi)}
        extendSymetrically(ds,'omega')
        
    es()
    import matplotlib.pyplot as plt
    hiddenLayers=[122,122,122,122]
    maxSteps=1500
    model_dir='/tmp/test-model/'
    chi=np.linspace(-0.5,1.5,20, dtype=np.float32)    
    om=omega(chi)

    plt.show()
    datasetS={'chi':chi,'omega':om}
    modelS=Model(datasetS, hiddenLayers, maxSteps,model_dir+'S/')
    # extend by reflection
    X0=chi[-1]
    ihc=chi[::-1]-X0
    ihc=-ihc+X0
    ihc=np.append(chi,ihc)
    
    Y0=om[-1]
    mo=om[::-1]-Y0
    mo=-mo+Y0
    mo=np.append(om,mo)
    
    datasetR={'chi':ihc,'omega':mo}    
    modelR=Model(datasetR, hiddenLayers, maxSteps,model_dir+'R/')
    from myInputFn import MyInputFn
    input_fn_S = MyInputFn(modelS.scaler,
            {'chi': np.linspace(-0.5,1.5,200, dtype=np.float32)})                                
    predictS=modelS.scaler.rescaleAs('omega',np.asarray(list(modelS.directRegressor.predict_scores(input_fn=input_fn_S(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale
    input_fn_R = MyInputFn(modelR.scaler,
                           {'chi': np.linspace(-0.5,1.5,200, dtype=np.float32)})                                    
    predictR=modelR.scaler.rescaleAs('omega',np.asarray(list(modelR.directRegressor.predict_scores(input_fn=input_fn_R(num_epochs=1,shuffle=False))))) # predicted is normalized now rescale

    plt.plot(input_fn_R.x['chi'],predictR, 'r+')
    plt.plot(input_fn_S.x['chi'],predictS, 'b+')
    plt.plot(chi, om, 'bo')
    plt.plot(ihc, mo, 'g-')
    # plt.plot(ihc, mo, 'g+')
    plt.show()
    
# test()
    
    
