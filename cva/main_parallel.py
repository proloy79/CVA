import ycmodel.hjm as h
import pandas as pd
import os
import config as cfg
import numpy as np
import datetime as dt
import multiprocessing
import mde.curve as dc
import default_prob as pbdf
import irs_exposure as irse
from functools import reduce
  
#regressDt='2013-2-12'
regressDt=dt.datetime(2013,2,12)
#refDt='2017-12-27'
refDt=dt.datetime(2017,12,27)

dtFormat='%Y%m%d'
filePrefix = regressDt.strftime(dtFormat) + '_' + refDt.strftime(dtFormat)
#filePrefix = regressDt + '_' + refDt
outFile = os.path.join(cfg.dataDir, filePrefix + '_fitted-outputs.xlsx')

try:
    os.remove(outFile)
except OSError:
    pass

hazardRate=.035
swapTenor=5
notional=1
freq=0.5
swapRate=0.015
eeTenors = np.arange(0, swapTenor+freq, freq)

def run_cva_worker(processNo, tasks, results, hjmModel,**params): 
    print('running process no : ', processNo)
    
    termExps = params['eeTenors']
    for i in tasks:    
        print('task id : ', i)
        simFwds = hjmModel.simulate_fwd_crv(params['swapTenor'])
        print(simFwds)
        irsExp = irse.irsexposure(params['notional'], params['swapTenor'], params['freq'], 
                             params['hazardRate'], params['swapRate'], simFwds, 
                             params['discCrv'], params['probDflt'])
        
        termExps = termExps + irsExp.calc_term_exposures()
    
    results[processNo] = termExps
    
    return

def calc_cva(eePerTerm, tenors, discCrv):
    cva=0.
    for i in np.arange(1, range(len(tenors))):
        df = discCrv.get_term_df(tenors[i-1], tenors[i])
        cva = cva + eePerTerm.iloc[i-1]*df
    
    return cva

writer = pd.ExcelWriter(outFile)
hm = h.hjm_model(regressDt, refDt, writer)
hm.calibrate_hjm()
writer.save()

discCrv = dc.disc_crv(refDt)
probDflt = pbdf.defaultprob(hazardRate, discCrv.tenors)
    
if __name__ == '__main__':                   
    params = {'discCrv':discCrv, 'probDflt':probDflt, 'swapTenor':swapTenor,
          'notional':notional, 'freq':freq, 'hazardRate' : hazardRate,
          'swapRate': swapRate, 'eeTenors' : eeTenors}}
    
    # Define IPC manager
    #manager = multiprocessing.Manager()    
    numProcesses = os.cpu_count()  
    results = np.zeros(numProcesses)    
    chunkSize = cfg.noSimSteps/numProcesses    
    #pool = multiprocessing.Pool(processes=numProcesses)  
    processes = []
    taskIds = np.arange(cfg.noSimSteps)
    
    # Initiate the worker processes
    for i in range(numProcesses):    
        fromId=i*chunkSize        
        exclusiveToId=fromId + chunkSize
        exclusiveToId = exclusiveToId if (cfg.noSimSteps-exclusiveToId > chunkSize) else exclusiveToId + cfg.noSimSteps-exclusiveToId
        tasks = np.arange(fromId, exclusiveToId)
        # Create the process, and connect it to the worker function
        newProcess = multiprocessing.Process(target=run_cva_worker, args=(i,tasks,results,hm,params))    
        # Add new process to the list of processes
        processes.append(newProcess)    
        # Start the process
        newProcess.start()
   
    #wait for all the threads to finish
    map(lambda x: x.join(), processes)
    EE = reduce(lambda x,y: x+y, results)/cfg.noSimSteps
    eeTenors = eeTenors = np.arange(0, swapTenor+freq, freq)
    cva = calc_cva(EE, eeTenors, discCrv)
    
    print(cva)
    
    print('\n\n######## done #######')