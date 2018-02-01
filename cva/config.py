import os

dataDir = os.path.realpath("./../data")
resultsDir = os.path.join(dataDir,"results")

coverageLevel=0.92          
sdeTs = 0.01
rawDataFile=os.path.realpath('./../data/USTREASURY-YIELD.csv')
serializeToFile=True
plotGraph=True

noSimSteps=100