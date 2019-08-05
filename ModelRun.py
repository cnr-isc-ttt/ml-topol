import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import argparse
import sys
import time
from sklearn import preprocessing
parser = argparse.ArgumentParser()
FLAGS = None
TYPE=np.float32
def main(_):
    from Model import padding
    # mode from 1 ... ncol-1 , trend next col NEW last col gamma
    def readData(fileName):
        def mkarray(x):
            return np.array([x],dtype=TYPE)
        infile=open(fileName) if len(fileName) else sys.stdin 
        dataRaw=np.array([ np.array([mkarray(token) for token in line.split()]) for line in infile.readlines()])
        chi=dataRaw[:, 0, None].reshape(-1) 
        indexes=chi>0.0
        datasetPositiveChi=dataRaw[indexes,:]
        indexes=chi<0.0
        datasetNegativeChi=dataRaw[indexes,:]
        
        return datasetPositiveChi,datasetNegativeChi
    class SelectMode:
        def __init__(self,padding):
            self.padding=padding
        def __call__(self,data, mode, trendSelected):    
            ds=self._selectMode(data, mode, trendSelected)
            if not self.padding:
                return ds
            
            gamma=ds['gamma']
            u,idx=np.unique(gamma, True) # values and indexes returned
            s=None
            chiOriginal=ds['chi']
            omegaOriginal=ds['omega']
            chiPadded=[]
            omegaPadded=[]
            gammaPadded=[] # this is not really padded just corrected for size
            for x in idx:
                if s is not None:
                    chi=chiOriginal[s:x]
                    chi=padding(chi)
                    gm=np.zeros_like(chi)
                    gm[:]=gamma[s] # enlarge gamma and fill with correct value
                    gammaPadded.append(gm)
                    chiPadded.append(chi)
                    omega=omegaOriginal[s:x]
                    omegaPadded.append(padding(omega))
                s=x
            # repeat one last time
            chi=chiOriginal[s:]
            chi=padding(chi)
            gm=np.zeros_like(chi)
            gm[:]=gamma[s] 
            gammaPadded.append(gm)
            chiPadded.append(chi)
            omega=omegaOriginal[s:]
            omegaPadded.append(padding(omega))
            gammaPadded=np.array(gammaPadded)
            gammaPadded=gammaPadded.reshape(-1)            
            chiPadded=np.array(chiPadded)
            chiPadded=chiPadded.reshape(-1)
            omegaPadded=np.array(omegaPadded)
            omegaPadded=omegaPadded.reshape(-1)
            fp=open('padded.chi.dat','ab')
            np.savetxt(fp,chiPadded)
            fp.close()
            fp=open('padded.om.dat','ab')
            np.savetxt(fp,omegaPadded)
            fp.close()
            print 'saved'
            chiPadded=np.array(chiPadded).reshape(-1)
            omegaPadded=np.array(omegaPadded).reshape(-1)
            gammaPadded=np.array(gammaPadded).reshape(-1)
            return {'chi':chiPadded,'omega':omegaPadded,'gamma':gammaPadded}
        
        
        def _selectMode(self,data, mode, trendSelected):    
            modeCol=1+(mode-1)*2
            # indexes=[len(rec)>(modeCol+1) for rec in dataRaw]
            # data=dataRaw[indexes]
            chi=data[:, 0, None] 
            omega=data[:, modeCol ,None] 
            gamma=data[:, -1, None] # last column is gamma
            trend=data[:, (modeCol+1) ,None] # slope field follows omega field
            trend=np.sign(trend)
            indexes=trend==trendSelected
            return {'chi':chi[indexes],'omega':omega[indexes], 'gamma':gamma[indexes]}        
        
    # prepare from flags
    hiddenLayers= eval('['+FLAGS.hiddenLayers+']')
    model_dir=FLAGS.export+'/'+FLAGS.hiddenLayers.replace(',','-')+'/'
    maxSteps = eval('int('+FLAGS.maxSteps+')')
    omegaStep=eval('float('+FLAGS.omegaStep+')')
    outfile=sys.stdout if FLAGS.output=='' else open(FLAGS.output,'w')
    outfile.write('# data: '+ FLAGS.data +"\n")
    outfile.write('# hiddenLayers: ['+ FLAGS.hiddenLayers+']' +"\n")
    outfile.write('# maxSteps: '+ FLAGS.maxSteps +"\n")
    outfile.write('# optimizer: '+ FLAGS.opt +"\n")
    run_metadata=tf.RunMetadata()
    def comment(): # not working
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # chrome://tracing
        # https://github.com/tensorflow/tensorflow/issues/1824#issuecomment-225754659
        # https://www.tensorflow.org/performance/performance_guide
        import os
        try:
            os.stat(model_dir)
        except:
            os.mkdir(model_dir)   
        trace_file = open(model_dir+'timeline.ctf.json', 'w')
    debug= True if FLAGS.debug else False
    if debug:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
    optimizer=FLAGS.opt
    if(FLAGS.opt=='grad'):
        model_dir=model_dir+'grad/'
    elif(FLAGS.opt=='prox'):
        model_dir=model_dir+'prox/'
    else:
        model_dir=model_dir # +'adam/'
        
    datasetPositiveChi,datasetNegativeChi=readData(FLAGS.data)
    padded=True if FLAGS.pad else False
    selectMode=SelectMode(padded)
    from Model import Model
    model={}
    modeRange= eval('range('+FLAGS.modeRange+')' ) if ',' in FLAGS.modeRange else [int(FLAGS.modeRange)]  # else single mode value    
    # create seperate models for each mode & trend with extra gamma feature 
    
    for mode in modeRange:
        for trend in range(-1,2,2):
            nm='p_'+str(mode)+'_'+str(trend)
            m=Model(selectMode(datasetPositiveChi, mode, trend), hiddenLayers, maxSteps, model_dir+nm, optimizer, debug) # positive only
            # trace_file.write(trace.generate_chrome_trace_format())
            if m.ok :
                model[nm]=m
            nm='n_'+str(mode)+'_'+str(trend)
            m=Model(selectMode(datasetNegativeChi, mode, trend), hiddenLayers, maxSteps, model_dir+nm, optimizer, debug)
            
            if m.ok :
                model[nm]=m
    # run models one for each mode & trend with extra gamma feature     
    
    from myInputFn import MyInputFn
    gammaValues= eval('np.linspace('+FLAGS.gammaRange+', dtype=TYPE)' ) if ',' in FLAGS.gammaRange else [TYPE(FLAGS.gammaRange)]  # else single gamma value    
    omegaTarget= eval('np.linspace('+FLAGS.omegaRange+', dtype=TYPE)' ) if ',' in FLAGS.omegaRange else None  # else auto 
    if omegaTarget is not None:
        outfile.write('# omegaRange: '+ FLAGS.omegaRange +"\n")
    trendFilter =False if FLAGS.trendFilter_off else True
    if trendFilter:
        outfile.write('# trendFilter ON' +"\n")
    
    for mode in modeRange:
        for trend in range(-1,2,2):
            for s in ['p','n']:
                nm=s+'_'+str(mode)+'_'+str(trend)
                if nm in model:
                    for gammaValue in gammaValues:
                        if omegaTarget is None:
                            omegaMin,omegaMax=model[nm].omegaRange(gammaValue)  
                            omegaPts=abs(int((omegaMin-omegaMax)/omegaStep))
                            omegaSweep=np.linspace(omegaMin,omegaMax, omegaPts)
                        else:
                            omegaSweep=np.copy(omegaTarget)
                        # LP cleanup - do a small step in omega to test trend
                        if trendFilter:
                            deltaOmega=np.abs(omegaSweep[1]-omegaSweep[0])*1e-3
                            tmp=np.append(omegaSweep, omegaSweep)
                            tmp[::2]=omegaSweep[:]
                            tmp[1::2]=omegaSweep[:]+deltaOmega
                            omegaSweep=tmp
                        print "# model: "+nm+" gammaValue: "+str(gammaValue)
                        gammaSweep=np.zeros_like(omegaSweep)
                        gammaSweep[:]=gammaValue
                        omegaSweep_input_fn_Inverse = MyInputFn(model[nm].scaler, # not great
                            {'omega': omegaSweep, 'gamma': gammaSweep})                                
                        chiPredicted,omegaPredicted,deviation=model[nm].SelfConsistantCycle(omegaSweep_input_fn_Inverse) # still need to do lp clean up 
                        if chiPredicted is not None:
                            if trendFilter:
                                trendPredicted=np.zeros_like(chiPredicted)
                                trendPredicted[:-1]=(omegaPredicted[1:]-omegaPredicted[:-1])/(chiPredicted[1:]-chiPredicted[:-1])
                                trendPredicted[-1]=trendPredicted[-2] # assume trend on last
                                trendPredicted=np.sign(trendPredicted)
                                goodTrendIdx= trendPredicted==trend
                                goodTrendIdx[1::2]=False # eliminate all deltaOmega
                            else:
                                goodTrendIdx=range(len(chiPredicted))
                            outfile.write("# "+nm+": omegaSC\tdeviation\tchi\tgamma\ttrend\tmode\n")
                            output(outfile,omegaPredicted[goodTrendIdx], deviation[goodTrendIdx], chiPredicted[goodTrendIdx], gammaValue, trend, mode)
                
def output(outfile, omegaSC, deviation,chi,gamma,trend, mode):

    for i in range(len(omegaSC)):
        outfile.write(str(omegaSC[i])+"\t"+str(deviation[i])+"\t"+str(chi[i])+"\t"+str(gamma)+"\t"+str(trend)+"\t"+str(mode)+"\n")
    outfile.write("\n\n")
if __name__ == '__main__':
    
    parser.add_argument('--maxSteps', type=str, default='25000',
                        help='maxSteps [25000] int ')   

    parser.add_argument('--omegaStep', type=str, default='0.00002',
                        help='omegaStep [0.00002] float ')   

    
    parser.add_argument('--hiddenLayers', type=str, default='131,131,131,131,131',
                        help='hiddenLayers [131,131,131,131,131] e.g. 12,12,12,12 as in list')   
        
    parser.add_argument('--data', type=str, default='',
                      help='Input file [stdin]')

    parser.add_argument('--export', type=str, default='/tmp/model',
                        help='Export model to path [/tmp/model]')
    
    parser.add_argument('--gammaRange', type=str, default='0.10,0.20,11',
                        help='gamma range [0.10,0.20,11] e.g. 10,12,300 as in np.linspace')   
    
    parser.add_argument('--modeRange', type=str, default='1,7',
                        help='mode index range [1,7] e.g. 1,2 as in range')   
    
    parser.add_argument('--omegaRange', type=str, default='',
                        help='omega range [auto] e.g. 5.67,5.7,100 as in linspace ')   
    
    parser.add_argument('--output', type=str, default='',
                        help='filename [stdout]')   
    
    parser.add_argument('--opt', type=str, default='',
                        help='optimizer [adam] e.g. grad | prox ')   

    parser.add_argument('--pad', action='store_true',
                        help='extend by padding')   
    
    parser.add_argument('--debug', action='store_true',
                        help='Fixed random seed and one CPU no GPU')   
    
    parser.add_argument('--trendFilter-off', action='store_true',
                        help='Trend filter off')   
    
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    
