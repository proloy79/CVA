import numpy as np
import pandas as pd
import math
import bisect

class defaultprob:
    def __init__(self, hazardRate, tenors):
        self.tenors = tenors
        self.hazardRate = hazardRate
        pdAtTenor = list(map(lambda t: math.exp(-1* self.hazardRate * t), tenors))
        #self.probDefault = [-1* pdAtTenor[0]] + list(map(lambda i,pdAtTenor=pdAtTenor: pdAtTenor[i-1] - pdAtTenor[i], pd.arange(1,len(tenors))))
        probDef = list(map(lambda i,pdAtTenor=pdAtTenor: pdAtTenor[i-1] - pdAtTenor[i], 
                           np.arange(1,len(tenors))))
        self.probDefault= pd.Series([0] + probDef, index=tenors)
    
    def get_pd(self, t):
        #as the pd's are valid for a term, if t falls inbetwen t1 and t2
        #return the value at t1                
        if t in self.tenors:            
            return self.probDefault[t]
        else:
            index = bisect.bisect(self.tenors, t)            
            return self.probDefault.iloc[index-1] 