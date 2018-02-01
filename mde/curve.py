import pandas as pd
import numpy as np
import math
import bisect
import os
#import config as cfg

class disc_crv:
    def __init__(self,refDt):        
        dataFile = os.path.join(os.path.abspath('../data'), 'OIS daily data current month.xlsx')   
               
#        fwds = pd.read_excel(dataFile,sheetname='1. fwd curve')        
#        fwds = fwds.set_index('Date')        
#        #fwds = [1, *np.transpose(fwds[refDt:refDt].values)[0:,0]]        
#        fwds = np.append([1], np.transpose(fwds[refDt:refDt].values)[0:,0])
        #the fwd data is inconsistent with the spots, better to derive directly from the spots
        spt = pd.read_excel(dataFile,sheetname='2. spot curve')        
        spt = spt.set_index('Date')                                
        spts = np.append([0], np.transpose(spt[refDt:refDt].values)[0:,0])  
        fwds = np.ones(len(spts))               
        
        self.tenors = np.append([0],spt.columns)       
        
        for i in range(1, len(self.tenors)):
            self.tenors[i] = float(str(round(self.tenors[i], 2)))
            spts[i] = spts[i]/100.
            t1 = self.tenors[i-1]
            r1= spts[i-1]            
            t2 = self.tenors[i]                                                                       
            r2=  spts[i]                                               
            fwds[i] = (r2*t2 - r1*t1)/(t2-t1)  
        
        #print('OIS tenors  : \n', self.tenors)     
        
        fwdsDf = pd.Series(fwds, index=self.tenors)
        
        self.fwdCrv = fwd_crv(fwdsDf)
        self.spots = pd.Series(spts, index=self.tenors)
        self.dfs = pd.Series(np.zeros(len(self.tenors)), index=self.tenors)
        
        for i in range(len(self.tenors)):
            self.dfs.iloc[i] = math.exp(-1*self.spots.iloc[i]*self.tenors[i])
        
    #get the df as of today(T=0)     
    def get_df(self, t):
        if t == 0:
            return 1.
                 
        if t in self.tenors:
            return self.dfs[t]
            #spot = self.spots[t]
        else:
            index = bisect.bisect(self.tenors, t)
            
            #if period exceeds the max tenor then exptrapolate
            if index>len(self.tenors) :            
                t1 = self.tenors[index-2]
                t2 = self.tenors[index]                
            else:                        
                leftIdx = self.tenors[index-1]
                rightIdx = self.tenors[index]                
                t1 = self.tenors[leftIdx]
                t2 = self.tenors[rightIdx]                
            y1=  self.spots[t1]
            y2=  self.spots[t2]
            
            spot = (t-t1)/(t2-t1)*y2 + (t2-t)/(t2-t1)*y1
            
        return math.exp(-1* spot * t)
    
    #get the df for the term T1-T2
    def get_fwd_df(self,t1,t2):
        return self.fwdCrv.get_fwd_df(t1,t2)
         
    def get_term_df(self, t1, t2):
        df1= self.get_df(t1)
        df2= self.get_df(t2)
        
        return (df1+df2)/2
    
class fwd_crv:
    def __init__(self,instFwds):
        self.instFwds = instFwds                
        tenors = np.copy(instFwds.index)        
        self.fwdTenors = []
        
        fwds=np.ones(len(tenors))
        
        for i in range(len(tenors)):
            self.fwdTenors.append(float(str(round(tenors[i], 2))))
            fwds[i] = fwds[i]/100
        
        #print('hjm fed tenors : \n' ,self.fwdTenors)    
        
        if tenors[0] != 0:
            self.fwdTenors = np.append([0], self.fwdTenors)            
            self.instFwds = pd.Series(np.append([1.], instFwds.values),index=self.fwdTenors)
        
        rts = np.zeros(len(self.fwdTenors))           

        for i in range(1,len(self.fwdTenors)): 
            t1 = self.fwdTenors[i-1]            
            t2 = self.fwdTenors[i]
            r1=rts[i-1]
            f2=self.instFwds[t2]
                     
            rts[i] = (f2*(t2-t1) + r1*t1)/t2
        
        self.rates = np.copy(rts)        
        
    def insert_new_tenors(self, newTenors):
        tnrs=np.zeros(len(newTenors))
        for i in range(len(newTenors)):
            tnrs[i] = float(str(round(newTenors[i], 2)))
         
        #print('hjm fed NEW tenors : \n' ,tnrs)
        tenors = sorted(set().union(np.copy(self.fwdTenors), tnrs))                
        rates = np.zeros(len(tenors))        
        fwds = np.ones(len(tenors)) 
        npNdArrTnr = np.ones(len(tenors))
        
        
        #for i in range(len(tenors)):
        #    tenors[i] = round(tenors[i], 2)
                        
        for i in range(len(tenors)):
            npNdArrTnr[i] = tenors[i]
            f,r = self.get_fwd(tenors[i])
            rates[i] = r
            fwds[i] = f
        #print(type(npNdArrTnr))
        self.rates = rates
        self.fwdTenors = npNdArrTnr
        self.instFwds=pd.Series(fwds,index=self.fwdTenors)

    def get_fwd(self, t):
        #t = round(t,2)
        if t == 0:
            return 1.,0.        
        elif t in self.fwdTenors:
            if(type(self.fwdTenors) is list):
                return self.instFwds[t],self.rates[self.fwdTenors.index(t)]            
            return self.instFwds[t],self.rates[np.where(self.fwdTenors==t)[0].item()]            
        else:
            #print(t)
            index = bisect.bisect(self.fwdTenors, t)
            
            #if period exceeds the max tenor then exptrapolate
            if index>len(self.fwdTenors) :            
                t1 = self.fwdTenors[index-2] 
                r1= self.rates[index-2]
            else:                        
                t1 = self.fwdTenors[index-1]
                r1= self.rates[index-1]
                
            t2 = self.fwdTenors[index]                                                                       
            r2=  self.rates[index]                               
            r = (t-t1)/(t2-t1)*r2 + (t2-t)/(t2-t1)*r1
            f = (r*t - r1*t1)/(t-t1)
            #print(t1,t,r1,r,f)
            return f,r
        
    def get_fwd_df(self,t1,t2):        
        idx1 = np.where(self.fwdTenors==t1)[0].item()
        idx2 = np.where(self.fwdTenors==t2)[0].item()
        
        df =1.
        for i in range(idx1,idx2):
            #print(i)
            dt = self.fwdTenors[i+1] - self.fwdTenors[i]
            df = df * math.exp(-1. * self.get_fwd(self.fwdTenors[i])[0] * dt) 
                    
        return df