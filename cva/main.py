import ycmodel.hjm as h
import pandas as pd
import os
import config as cfg
import numpy as np
import datetime as dt
import mde.curve as dc
import default_prob as pbdf
import irs_exposure as irse
import matplotlib.pyplot as plt
import time

startTime = time.time()  

################ user inputs starts ################
regressDt=dt.datetime(2013,2,12)
refDt=dt.datetime(2017,12,27)
hazardRate=0.05
swapMaturity=5
notional=1
freq=0.5
swapRate=2.5
recoveryRate=0.4
################ user inputs ends ###################

dtFormat='%Y%m%d'
filePrefix = regressDt.strftime(dtFormat) + '_' + refDt.strftime(dtFormat) + '_' + str(cfg.noSimSteps)
outputFile = os.path.join(cfg.resultsDir, filePrefix + '_fitted-outputs.xlsx')
inputFile = os.path.join(cfg.resultsDir, filePrefix + '_fitted-inputs.xlsx')

def remove_file(filePath):
    try:
        os.remove(filePath)    
    except OSError:
        pass
    
remove_file(outputFile)
remove_file(inputFile)
eeTenors = np.arange(0, swapMaturity+freq, freq)
    
def run_cva_worker(processNo, tasks, hjmModel,**params): 
    #print('running process no : ', processNo)
    
    termExps = np.zeros(len(params['eeTenors'])-1)
    for i in tasks:    
        print('running task id : ', i)
        
        simFwds = hjmModel.simulate_fwd_crv(params['swapMaturity'])  
        simFwds.insert_new_tenors(params['eeTenors'])
        
        irsExp = irse.irsexposure(params['notional'], params['swapMaturity'], 
                                  params['freq'], params['swapRate'], simFwds, 
                                  params['discCrv'])
        
        exposures = irsExp.calc_term_exposures()    
        
        if cfg.plotGraph:
            f1 = plt.figure(1)
            plt.plot(exposures)
            plt.xlabel('tenors')
            plt.ylabel('exposures')
            f1.savefig(os.path.join(cfg.resultsDir, str(cfg.noSimSteps)+'_Exposures.png'))
            
            f2 = plt.figure(2)
            plt.plot(simFwds.instFwds)
            plt.xlabel('tenors')
            plt.ylabel('simulated fwds')
            f2.savefig(os.path.join(cfg.resultsDir, str(cfg.noSimSteps)+'_SimulatedFwds.png'))
            
        termExps = np.add(termExps, exposures.values[0:-1])        
    
    #print('\nEE : ', termExps)
    #results[processNo] = termExps
    return termExps
    

def calc_cva(eePerTerm, tenors, discCrv,probDflt,rr):
    cva=0.
    if cfg.plotGraph:
            f3 = plt.figure(3)
            plt.plot(eePerTerm)
            plt.xlabel('tenors')
            plt.ylabel('EE')
            f3.savefig(os.path.join(cfg.resultsDir, str(cfg.noSimSteps)+'_EE.png'))
            
    for i in np.arange(1, len(tenors)-1):
        df = discCrv.get_term_df(tenors[i-1], tenors[i])
        cva = cva + eePerTerm[i-1]*df*(1-rr)*probDflt.get_pd(tenors[i])
    
    return cva

#set up the market data env
inWriter = pd.ExcelWriter(inputFile)
hm = h.hjm_model(regressDt, refDt, inWriter)
hm.calibrate_hjm()

discCrv = dc.disc_crv(refDt)
probDflt = pbdf.defaultprob(hazardRate, eeTenors)

if (cfg.serializeToFile):
    pd.Series(probDflt.probDefault).to_excel(inWriter, 'PD') 
    dcDf = pd.DataFrame(np.zeros([len(discCrv.tenors),3]), columns=['Fwd','Spot','Df'], index=discCrv.tenors)  
    dcDf['Fwd'] = discCrv.fwdCrv.instFwds.values
    dcDf['Spot'] = discCrv.spots.values
    dcDf['Df'] = discCrv.dfs.values
    dcDf.to_excel(inWriter, 'DiscCrv')

inWriter.save()

#cva calculations start from here
outWriter = pd.ExcelWriter(outputFile)                   
params = {'discCrv':discCrv, 'swapMaturity':swapMaturity,
          'notional':notional, 'freq':freq, 
          'swapRate': swapRate, 'eeTenors' : eeTenors}

EE = run_cva_worker(0,
               np.arange(0,cfg.noSimSteps),
               hm,
               **params)   

EE = EE/cfg.noSimSteps
cva = calc_cva(EE, eeTenors, discCrv, probDflt, recoveryRate)

netTime=divmod((time.time() - startTime),60)
netTime = '{0}min {1}sec'.format(round(netTime[0],2), round(netTime[1],2))

if (cfg.serializeToFile):
    eeDf = pd.DataFrame(np.zeros([len(EE),2]), columns=['EE','NoOfSimPaths'], index=eeTenors[1:])  
    eeDf['EE'] = EE
    eeDf['NoOfSimPaths'] = np.full(len(EE),cfg.noSimSteps)
    eeDf.to_excel(outWriter, 'EE') 
    
    cvaDf = pd.DataFrame(np.zeros([1,3]), columns=['NoSimSteps','CVA','TimeTaken'],index=None) 
    cvaDf['NoSimSteps'] = [cfg.noSimSteps]
    cvaDf['CVA'] = [cva]
    cvaDf['TimeTaken'] = [netTime]
    cvaDf.to_excel(outWriter, 'cva')
    
outWriter.save()

print('\ncva : ', cva)
print("\n******** ",cfg.noSimSteps," simulations : ",netTime," *****")
print('\n\n######## done #######')
      