import pandas as pd
import numpy as np
import math
import config as cfg
from scipy.integrate import quad
import random
import sys
import mde.curve as crv

noOfPCAs=-1

def load_fwd_rates(data):          
    rowCount =  len(data)               
    tenors = [float(i) for i in list(data)]
    fwds = np.zeros([rowCount, len(tenors)])
        
    for i in range(rowCount):                
        for j in range(len(tenors)):        
            if(j==0):                
                zc = math.exp(-1*data.iloc[i,0] * tenors[j])
                fwds[i, 0] = -math.log(sys.float_info.min if zc==0 else zc)/tenors[0]
            else: 
                zcT = math.exp(-1*data.iloc[i,j] * tenors[j])
                zcT_1 = math.exp(-1*data.iloc[i,j-1] * tenors[j-1])
                fwds[i,j] = -(math.log((sys.float_info.min if zcT==0 else zcT)/
                        (sys.float_info.min if zcT_1==0 else zcT_1))
                        /(tenors[j] - tenors[j-1]))  
                
    return pd.DataFrame(fwds, columns=tenors)  

def calc_excess(data):          
    rowCount =  len(data)               
    tenors = list(data)
    noOfTenors= len(tenors)
    excessRtn = np.zeros([rowCount-1, noOfTenors])
            
    for j in range(noOfTenors):
        for i in range(1, rowCount):
            excessRtn[i-1, j] = data.iloc[i,j] - data.iloc[i-1,j]                       
    
    return pd.DataFrame(excessRtn, columns=tenors) 

def gen_cov_matrix(data):
    N = len(data)        
    covariance = 1/N * np.dot(np.transpose(data),data) * 252. / 10000.
    return covariance

def strip_pca_vecs(eigenValues, eigenVecs, tenors):        
    rowCount = len(tenors)            
    
    sortedEigenIdxs = np.flip(sorted(range(len(eigenValues)), key=lambda i,
                        eigenValues=eigenValues: eigenValues.iloc[i]), 0)
    
    netLambda = eigenValues.sum()
    cumPercent = np.zeros(rowCount) 
    cumPercent[0] = eigenValues.iloc[sortedEigenIdxs[0]]/netLambda
    
    global noOfPCAs
    for i in range(1,rowCount):
        cumPercent[i] = cumPercent[i-1] +eigenValues.iloc[sortedEigenIdxs[i]]/netLambda
        if cumPercent[i] >= cfg.coverageLevel:
            noOfPCAs = i+1
            break
        
    #picking the largest components
    largestEigenIdxs = np.flip(sorted(range(len(eigenValues)), key=lambda i,
                        eigenValues=eigenValues: eigenValues.iloc[i])[-1*noOfPCAs:], 0)
    
    vols = pd.DataFrame(np.zeros([rowCount, noOfPCAs]), index=tenors) 
    
    for j in range(len(largestEigenIdxs)):
        #print(largestEigenIdxs[j])
        vols.iloc[0:,j] = eigenVecs.iloc[0:,largestEigenIdxs[j]]     
        
    return vols

def fit_vol(tau, volParams, pcIdx):
        if(pcIdx==0):            
            return volParams[0]
    
        params = volParams[pcIdx]
        return params[3] + tau*params[2] + tau**2 * params[1] + tau**3 * params[0]

class hjm_model:
    def __init__(self, startDt, endDt, logWriter):                             
        self.startDt = startDt
        self.endDt = endDt
        self.logWriter = logWriter
        
    def read_yc_tbl(self):          
        df = pd.read_csv(filepath_or_buffer=cfg.rawDataFile,parse_dates=['Date'])         
        df.sort_values(by=['Date'])
        df.set_index('Date', inplace=True)         
        data = df[self.startDt:self.endDt:]
        data = pd.DataFrame(data.values,columns=data.columns,index=np.arange(0,data.shape[0]),dtype=float)
        data = data.apply(pd.to_numeric, errors='ignore')
        #print(data)
        return data
    
    def calc_eigen_values(self, covMatrix):          
        eigenValues,eigenVecs = np.linalg.eig(np.matrix(covMatrix))
        self.eigenValues = pd.Series(eigenValues,index=self.tenors)
        #the computed eigenvecs donot match the macro, recomputing
        self.eigenVecs = pd.DataFrame(eigenVecs,columns=self.tenors,index=self.tenors)
        for i in range(len(self.tenors)):
            self.eigenVecs.iloc[:,i] = self.eigenVecs.iloc[:,i]*math.sqrt(self.eigenValues.iloc[i])

    def calibrate_hjm(self):
        data = self.read_yc_tbl()
        self.tenors = [round(float(i),2) for i in list(data)]
        fwds = load_fwd_rates(data)
        self.todays_fwds = fwds.iloc[-1,0:]
        excessRtn = calc_excess(fwds)        
        
        #function shows 1 basis pont difference in the std dev
        #covMatrix = gen_cov_matrix(excessRtn)                   
        
        #using the numpy cov matches the sample covariance in s/s
        covMatrix = np.cov(np.transpose(excessRtn))  *252/10000                  
        #print(covMatrix)              
        self.calc_eigen_values(covMatrix)
        self.pcVecs = strip_pca_vecs(self.eigenValues, self.eigenVecs, self.tenors)
        self.gen_vol_fit_params()
        self.gen_fitted_pc_vols()
        self.gen_drifts()
        
        if (cfg.serializeToFile):
            data.to_excel(self.logWriter, 'Rates')
            fwds.to_excel(self.logWriter, 'Fwds')                 
            excessRtn.to_excel(self.logWriter, 'ExcessRtn')                          
            pd.DataFrame(covMatrix,index=self.tenors,columns=self.tenors).to_excel(self.logWriter, 'Covariance')            
            self.eigenValues.to_excel(self.logWriter, 'EigenValues')
            self.eigenVecs.to_excel(self.logWriter, 'EigenVecs')
            self.pcVecs.to_excel(self.logWriter, 'PcVecs')
            self.volFitParams.to_excel(self.logWriter, 'VolFitParams')
            self.fittedPcVols.to_excel(self.logWriter, 'FittedPcEigVols')
            self.drifts.to_excel(self.logWriter, 'Drifts')            
        
    def simulate_fwd_crv(self, tau): 
        dt = cfg.sdeTs
        noOfSteps = int(tau/dt)
        noOfTenors = len(self.tenors)
        
        #fwds = pd.DataFrame(np.zeros([noOfSteps, noOfTenors]), 
        #                    index=np.arange(noOfSteps), columns=self.tenors)                        
        fwds = pd.Series(self.todays_fwds.copy(),index=self.tenors)
        #print('\nbase fwds : \n',self.todays_fwds)
        #for tnrNo in range(noOfTenors):
        #    fwds.iloc[0,tnrNo] = self.todays_fwds.iloc[tnrNo]
                
        for stepNo in range(1,noOfSteps):
            rnd = np.zeros(noOfPCAs)
            
            for x in range(noOfPCAs):
                rnd[x] = random.gauss(0,1)
            #print(rnd)    
            prevFwds = fwds.copy()
            for tnrNo in range(noOfTenors):
                #prevFwd = fwds.iloc[stepNo -1, tnrNo]                
                nextTnrNo = (tnrNo-1) if (tnrNo==noOfTenors-1) else (tnrNo+1)                
                #prevFwdRight = fwds.iloc[stepNo -1, nextTnrNo]
                prevFwdRight = prevFwds.iloc[nextTnrNo]
                drift = self.drifts.iloc[tnrNo]
                
                volComp = 0.                
                for pcNo in range(noOfPCAs):
                    volComp = volComp + self.fittedPcVols.iloc[tnrNo,pcNo]*rnd[pcNo] 
                    
                fwds.iloc[tnrNo] = (prevFwds.iloc[tnrNo] + drift*dt + 
                             volComp * math.sqrt(dt) + 
                             (prevFwdRight - prevFwds.iloc[tnrNo])/
                             (self.tenors[nextTnrNo] - self.tenors[tnrNo])*dt)
                                        
                #print(stepNo, "  :  ",self.tenors[tnrNo], "  :  ", volComp,"  :  ",prevFwds.iloc[-1], "  :  ",fwds.iloc[-1])
        #if (cfg.serializeToFile):
        #    fwds.to_excel(self.logWriter, 'SimFwds')
        
        #return crv.fwd_crv(pd.Series(fwds.iloc[-1].values,index=self.tenors))
        #print('\nsimulated fwds :\n',fwds)
        return crv.fwd_crv(fwds)
    
    def gen_vol_fit_params(self):                 
        self.volFitParams = pd.Series(np.zeros(noOfPCAs),index=np.arange(0,noOfPCAs),dtype='object')                
        
        #for first pc use median as it provides for parallel shift
        self.volFitParams[0] = np.median(self.pcVecs.iloc[0:,0])
            
        for i in range(1,noOfPCAs):  
            try:
                self.volFitParams.iloc[i] = np.polyfit(self.tenors,self.pcVecs.iloc[0:,i],3)   
            except:
                print("Exception in gen_vol_fit_params :", sys.exc_info()[0])
                raise
    
    def gen_fitted_pc_vols(self):                 
        self.fittedPcVols = pd.DataFrame(np.zeros([len(self.tenors),noOfPCAs]),
                                         columns=np.arange(0,noOfPCAs), 
                                         index=self.tenors)                
        
        for tnrNo in range(len(self.tenors)):
            for i in range(noOfPCAs):  
                self.fittedPcVols.iloc[tnrNo,i] = fit_vol(self.tenors[tnrNo], self.volFitParams, i)
                
    def gen_drifts(self):                 
        self.drifts = pd.Series(np.zeros(len(self.tenors)),index=self.tenors)                
        
        for tnr in self.tenors:                        
            self.drifts[tnr] = self.calc_drift_at(tnr)
            
    #helper to calculate the drift at a particular tenor
    def calc_drift_at(self, tau):                 
        drift=0.
        dTau = cfg.sdeTs
        n=int(tau/dTau)
        #print('\n\n','tau : ',tau)
        for i in range(noOfPCAs):            
            pcDrift = 0.5 * fit_vol(0, self.volFitParams, i)
            #print(0,pcDrift)
            for j in range(1, n):
                pcDrift = pcDrift + fit_vol(j*dTau, self.volFitParams, i)
                #print(j,pcDrift)
            
            pcDrift = pcDrift + 0.5 * fit_vol(tau, self.volFitParams, i)
            #print(tau,pcDrift)
            pcDrift = pcDrift * dTau
            #print(pcDrift)
            pcDrift = pcDrift * fit_vol(tau, self.volFitParams, i)
            #print(pcDrift)
            
            drift =drift + pcDrift
            #print('total for pc :',i,' : ',drift,'\n')
        #print('total for ',tau, ' : ',drift)
        return drift
         
        