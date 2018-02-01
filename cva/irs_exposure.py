import pandas as pd
import numpy as np
        
class irsexposure:
    def __init__(self, notional, swapTenor, freq,  swapRate, simFwdCrv, discCrv):        
        self.notional = notional                
        self.freq = freq           
        self.swapRate = swapRate          
        self.tenors = np.arange(0, swapTenor+freq, freq)
        simFwdCrv.insert_new_tenors(self.tenors)
        self.simFwdCrv = simFwdCrv
        self.discCrv = discCrv        
    
    def calc_term_exposures(self):
        noOfTnrs = len(self.tenors)        
        cashflows = np.zeros(noOfTnrs)
        e = np.zeros(noOfTnrs)
        exposures = np.zeros(noOfTnrs)
        
        #start from T1        
        cashflows[0] = 0.

        #print('\ntenor','fwd','swapRate','cashflow')             
        for i in range(1,noOfTnrs): 
            fwd = self.simFwdCrv.instFwds.iloc[i]
            cashflows[i] = self.notional*(fwd - self.swapRate)*self.freq        
            #print(self.tenors[i],fwd,self.swapRate,cashflows[i])
        
        #print(self.simFwdCrv.instFwds)
        
        for i in range(noOfTnrs-1):
            mtm = 0.            
            t1 = self.tenors[i]
            
            #print('\ntnr:',t1,'\n')
            
            for j in range(i+1,noOfTnrs):                
                t2 = self.tenors[j]                                 
                df = self.discCrv.get_fwd_df(t1, t2)                 
                mtm = mtm + cashflows[j] * df
                #print(t1,cashflows[j], df)
            
            e[i] = max(mtm,0)
            #print(t1,e[i-1])
            
        e[noOfTnrs-1] = 0
        
        #print('\n e : ',e)
        
        for i in range(noOfTnrs-1):
            #print(i,e[i], e[i+1])
            exposures[i] = (e[i] + e[i+1])/2.
        
        return pd.Series(exposures, index=self.tenors)
            
    
            