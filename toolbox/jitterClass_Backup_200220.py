# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 08:46:40 2020

@author: joerg_j
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from readTrc import Trc
import re
import math
from scipy.optimize import curve_fit

class analyseMeasurement:
    
    
    def __init__(self, rawdataFolder = None, campaignId = None, measurementId = None, diagnose = False):
        
        self.measurementId = measurementId
        self.diagnose = diagnose
        if(measurementId is not None):
            self.dataFolder = os.path.join(rawdataFolder, campaignId, measurementId)
        else:
            self.dataFolder = None
        
        self.data = None
        self.meta = None
        self.gauss = None
        self.thresA = {}
        self.thresR = {}
        self.delta = {}
        self.hist = {}
        self.result = {}
        
    
    def _gauss(self, x, A, mu = None, sigma = None):
        if(mu == None):
            A, mu, sigma = A
        return A * np.exp(-np.power((x - mu) / sigma, 2) / 2) / (sigma * math.sqrt(2 * math.pi))
 
    
    def read(self, traceName = "Trace", pulseMin = None, pulseMax = None, channels = ["C1", "C3"]):
        
        dataAll = []
        metaAll = []
        
        for channel in channels:
            
            data = []
            meta = []
            for fn in os.listdir(self.dataFolder):
                if(re.match(r"^" + channel + traceName + "[0-9]{5}\.trc$", fn)):
                    if(pulseMin is not None):
                        if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) < pulseMin:
                            continue
                    if(pulseMax is not None):
                        if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) >= pulseMax:
                            continue
                    trc = Trc()
                    trc.open(os.path.join(self.dataFolder, fn))
                    data.append(np.array((trc.x, trc.y)))
                    meta.append(trc.d)
            
            dataAll.append(data)
            metaAll.append(meta)
            
        self.data = np.array(dataAll)
        self.meta = np.array(metaAll)
        
        if(self.diagnose):
            print("read done")



    
    def showRawPulse(self, pulseNr = 0, areaMin = None, areaMax = None, xy = [0, 1], channelNrs = [0, 1], style = "-"):
        
        for channelNr in channelNrs:
            plt.plot(self.data[channelNr, pulseNr, xy[0], areaMin:areaMax], self.data[channelNr, pulseNr, xy[1], areaMin:areaMax], style)
        plt.show()
        
        if(self.diagnose):
            print("showRawPulse done")
        
    
    
    
    def fitGauss(self, threshold = 0.1, channelNrs = [0, 1], xy = [0, 1], gaussParams = None, diagnosePulse = 0):
        
        gaussAll = []
        
        for channelNr in channelNrs:
            gauss = []
            i = 0
            for pulse in self.data[channelNr]:
                
                #search maximum
                iPeak  = np.argmin(pulse, axis=1)[1]
                iLeft  = np.where(pulse[1, :iPeak] >= threshold * pulse[1, iPeak])[0][-1]
                iRight = np.where(pulse[1, iPeak:] >= threshold * pulse[1, iPeak])[0][0] + iPeak
                
                if(self.diagnose and i == diagnosePulse):
                    print("iPeak:", iPeak, "iLeft:", iLeft, "iRight:", iRight)
                    plt.plot(self.data[channelNr,0,0], self.data[channelNr,0,1])
                    plt.plot(self.data[channelNr,0,0,iPeak],self.data[channelNr,0,1,iPeak], "o")
                    plt.plot(self.data[channelNr,0,0,iLeft],self.data[channelNr,0,1,iLeft], "o")
                    plt.plot(self.data[channelNr,0,0,iRight - 1],self.data[channelNr,0,1,iRight - 1], "o")
                    plt.show()
                    plt.plot(self.data[channelNr,0,0,iLeft:iRight], self.data[channelNr,0,1,iLeft:iRight], "o-")
                    plt.show()
                
                if(gaussParams is None):
                    gaussParamsLoop = [1e-10, self.data[channelNr,0,0,iPeak], 5e-11]
                else:
                    gaussParamsLoop = gaussParams
                
                try:
                    fit_params, var_matrix = curve_fit(self._gauss, pulse[xy[0]][iLeft:iRight], pulse[xy[1]][iLeft:iRight], p0 = gaussParamsLoop)
                except:
                    print("failed.")
                    return None
                
                if(self.diagnose and i == diagnosePulse):
                    plt.plot(pulse[0],pulse[1])
                    xFit = np.linspace(pulse[0,iLeft], pulse[0,iRight], 50)
                    plt.plot(xFit, self._gauss(xFit, fit_params))
                    plt.show()
                    print("parameter:", fit_params, "[A, mu, sigma]")

                i += 1
            
                gauss.append([fit_params[0], fit_params[1], fit_params[2], iLeft, iPeak, iRight])
            gaussAll.append(gauss)
        
        self.gauss = np.array(gaussAll)
        
        if(self.diagnose):
            print("fitGauss done")

    
    
    
    def thresholdRelative(self):
        
        return None
    
    
    

    def thresholdAbsolute(self):
        
        return None
    
    
    
    def calculateDelta(self, saveTo, data1, data2):

        self.delta[saveTo] = data1 - data2
        
        if(self.diagnose):
            print("calculateDelta done")

    
    
    
    def histogram(self, takeFrom, saveTo, bins = 100, range = None):
        
        HistY, edges, plot = plt.hist(self.delta[takeFrom], bins, range = range)
        plt.show()
        plt.close()
        self.hist[saveTo] = ((edges[1:] + edges[:-1]) / 2, HistY, edges)
        
        if(self.diagnose):
            print("histogram done")
    
    
    
    def fitHist(self, takeFrom, saveTo, gaussParams = None, areaMin = None, areaMax = None):
        
        print(np.argmax(self.hist[takeFrom][1]))
        if(gaussParams is None):
            gaussParamsLoop = [1e-10, self.hist[takeFrom][0][np.argmax(self.hist[takeFrom][1])], 1e-12]
        else:
            gaussParamsLoop = gaussParams
        
        fit_params, var_matrix = curve_fit(self._gauss, self.hist[takeFrom][0][areaMin:areaMax], self.hist[takeFrom][1][areaMin:areaMax], p0 = gaussParamsLoop)
        #plt.close()
        plt.plot(self.hist[takeFrom][0], self.hist[takeFrom][1])
        plt.plot(self.hist[takeFrom][0], self._gauss(self.hist[takeFrom][0], fit_params))
        plt.show()
        print("parameter:", fit_params, "[A, mu, sigma]")
        
        self.result[saveTo] = fit_params
        
        if(self.diagnose):
            print("fitHist done")
    
        
        
        
    def exportDict(self):
        
        export = {}
        
        export["measurementId"] = self.measurementId
        export["dataFolder"] = self.dataFolder
        export["data"] = self.data
        export["meta"] = self.meta
        export["gauss"] = self.gauss
        export["thresA"] = self.thresA
        export["thresR"] = self.thresR
        export["delta"] = self.delta
        export["hist"] = self.hist
        export["result"] = self.result
        
        if(self.diagnose):
            print("exportDict done")
                
        return export
    
    
    def importDict(self, importDict):
        
        self.measurementId = importDict["measurementId"]
        self.dataFolder = importDict["dataFolder"]
        self.data = importDict["data"]
        self.meta = importDict["meta"]
        self.gauss = importDict["gauss"]
        self.thresR = importDict["thresR"]
        self.thresA = importDict["thresA"]
        self.delta = importDict["delta"]
        self.hist = importDict["hist"]
        self.result = importDict["result"]
        
        if(self.diagnose):
            print("importDict done")
    
    
    

