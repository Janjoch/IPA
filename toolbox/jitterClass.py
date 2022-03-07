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
    
    
    def __init__(self, rawdataFolder = None, campaignId = None, measurementId = None, signalAmplitude = None, diagnose = False):

        self.data = {}
        self.data["thres"]  = {}
        self.data["delta"]  = {}
        self.data["hist"]   = {}
        self.data["result"] = {}
        
        self.data["measurementId"] = measurementId
        self.data["signalAmplitude"] = signalAmplitude
        self.data["diagnose"] = diagnose
        if(measurementId is not None):
            self.data["dataFolder"] = os.path.join(rawdataFolder, campaignId, measurementId)
        else:
            self.data["dataFolder"] = None
        
        self.diagnose(diagnose)
        
        
    
    def diagnose(self, active = False, bigPlots = False, grid = True, figsize = (12,6)):
        
        self.data["diagnose"] = active
        
        self.data["plt"] = {}

        self.data["plt"]["bigPlots"] = bigPlots
        self.data["plt"]["grid"] = grid
        self.data["plt"]["figsize"] = figsize
    
   
    
    
    def preparePlot(self, xlabel = "", ylabel = ""):

        plt.close()
        
        if(self.data["plt"]["bigPlots"]):
            plt.figure(figsize=self.data["plt"]["figsize"])
            plt.rc("font", size=15)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(self.data["plt"]["grid"])
        
        
    
    
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
            for fn in os.listdir(self.data["dataFolder"]):
                if(re.match(r"^" + channel + traceName + "[0-9]{5}\.trc$", fn)):
                    if(pulseMin is not None):
                        if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) < pulseMin:
                            continue
                    if(pulseMax is not None):
                        if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) >= pulseMax:
                            continue
                    trc = Trc()
                    trc.open(os.path.join(self.data["dataFolder"], fn))
                    data.append(np.array((trc.x, trc.y)))
                    meta.append(trc.d)
            
            dataAll.append(data)
            metaAll.append(meta)
            
        self.data["raw"] = np.array(dataAll)
        self.data["meta"] = np.array(metaAll)
        
        if(self.data["diagnose"]):
            print("read done")



    
    def showRawPulse(self, pulseNr = 0, areaMin = None, areaMax = None, xy = [0, 1], channelNrs = [0, 1], style = "-"):
        
        self.preparePlot("t [s]", "U [V]")
        
        for channelNr in channelNrs:
            plt.plot(self.data["raw"][channelNr, pulseNr, xy[0], areaMin:areaMax], self.data["raw"][channelNr, pulseNr, xy[1], areaMin:areaMax], style)
        plt.show()
        
        if(self.data["diagnose"]):
            print("showRawPulse done")
        
    
    
    
    def fitGauss(self, threshold = 0.1, channelNrs = [0, 1], xy = [0, 1], gaussParams = None, diagnosePulse = 0):
        
        gaussAll = []
        
        for channelNr in channelNrs:
            gauss = []
            i = 0
            for pulse in self.data["raw"][channelNr]:
                
                #search maximum
                iPeak  = np.argmin(pulse, axis=1)[1]
                iLeft  = np.where(pulse[1, :iPeak] >= threshold * pulse[1, iPeak])[0][-1]
                iRight = np.where(pulse[1, iPeak:] >= threshold * pulse[1, iPeak])[0][0] + iPeak
                
                if(self.data["diagnose"] and i == diagnosePulse):
                    print("iPeak:", iPeak, "iLeft:", iLeft, "iRight:", iRight)
                    self.preparePlot("t [s]", "U [V]")
                    plt.plot(self.data["raw"][channelNr,0,0], self.data["raw"][channelNr,0,1])
                    plt.plot(self.data["raw"][channelNr,0,0,iPeak],self.data["raw"][channelNr,0,1,iPeak], "o")
                    plt.plot(self.data["raw"][channelNr,0,0,iLeft],self.data["raw"][channelNr,0,1,iLeft], "o")
                    plt.plot(self.data["raw"][channelNr,0,0,iRight - 1],self.data["raw"][channelNr,0,1,iRight - 1], "o")
                    plt.show()
                    self.preparePlot("t [s]", "U [V]")
                    plt.plot(self.data["raw"][channelNr,0,0,iLeft:iRight], self.data["raw"][channelNr,0,1,iLeft:iRight], "o-")
                    plt.show()
                
                if(gaussParams is None):
                    gaussParamsLoop = [1e-10, self.data["raw"][channelNr,0,0,iPeak], 5e-11]
                else:
                    gaussParamsLoop = gaussParams
                
                try:
                    fit_params, var_matrix = curve_fit(self._gauss, pulse[xy[0]][iLeft:iRight], pulse[xy[1]][iLeft:iRight], p0 = gaussParamsLoop)
                except:
                    print("failed.")
                    return None
                
                if(self.data["diagnose"] and i == diagnosePulse):
                    self.preparePlot("t [s]", "U [V]")
                    plt.plot(pulse[0],pulse[1])
                    xFit = np.linspace(pulse[0,iLeft], pulse[0,iRight], 50)
                    plt.plot(xFit, self._gauss(xFit, fit_params))
                    plt.show()
                    print("parameter:", fit_params, "[A, mu, sigma]")

                i += 1
            
                gauss.append([fit_params[0], fit_params[1], fit_params[2], iLeft, iPeak, iRight])
            gaussAll.append(gauss)
        
        self.data["gauss"] = np.array(gaussAll)
        
        if(self.data["diagnose"]):
            print("fitGauss done")

    
    
    
    def threshold(self, absolute, threshold, channelNrs = [0, 1], diagnosePulse = 0):
        
        thresAll = []
        
        for channelNr in channelNrs:
            thresF = []
            thresC = []
            thresR = []
            i = 0
            for pulse in self.data["raw"][channelNr]:
                
                #search maximum
                iPeak  = np.argmin(pulse, axis=1)[1]
                if(absolute):
                    yt = - threshold * self.data["signalAmplitude"]
                else:
                    yt = threshold * pulse[1, iPeak]
                iLeft  = np.where(pulse[1, :iPeak] >= yt)[0][-1]
                iRight = np.where(pulse[1, iPeak:] >= yt)[0][0] + iPeak - 1
                
                #left
                x1 = pulse[0, iLeft]
                x2 = pulse[0, iLeft + 1]
                y1 = pulse[1, iLeft]
                y2 = pulse[1, iLeft + 1]
                xt = (x2 - x1) * (yt - y1) / (y2 - y1) + x1
                
                if(self.data["diagnose"]):
                    if(diagnosePulse == i):
                        self.preparePlot("t [s]", "U [V]")
                        plt.plot(pulse[0], pulse[1])
                        plt.plot(x1, y1, "o")
                        plt.plot(x2, y2, "o")
                        plt.plot([pulse[0, 0], pulse[0, -1]], [yt, yt], "-")
                        plt.show()
                        
                        print("xt:", xt, "ps, yt", yt, "ps")
                
                thresF.append((xt, yt))
                
                xtl = xt
                ytl = yt

                #right
                x1 = pulse[0, iRight]
                x2 = pulse[0, iRight + 1]
                y1 = pulse[1, iRight]
                y2 = pulse[1, iRight + 1]
                xt = (x2 - x1) * (yt - y1) / (y2 - y1) + x1
                
                if(self.data["diagnose"]):
                    if(diagnosePulse == i):
                        self.preparePlot("t [s]", "U [V]")
                        plt.plot(pulse[0], pulse[1])
                        plt.plot(x1, y1, "o")
                        plt.plot(x2, y2, "o")
                        plt.plot([pulse[0, 0], pulse[0, -1]], [yt, yt], "-")
                        plt.show()
                        
                        print("xt:", xt, "ps, yt", yt, "ps")
                
                thresR.append((xt, yt))
                
                #center
                thresC.append(((xtl + xt) / 2, (ytl + yt) / 2))
                
                
                i += 1
                    
            thresAll.append((thresF, thresC, thresR))
            
        if(absolute):
            s = "Abs"
        else:
            s = "Rel"
        self.data["thres"]["thres{:d}".format(int(threshold * 100)) + s] = np.array(thresAll)
    

    def calculateDelta(self, saveTo, data1, data2):

        self.data["delta"][saveTo] = data1 - data2
        
        if(self.data["diagnose"]):
            print("calculateDelta done")

    
    
    
    def histogram(self, takeFrom, saveTo, bins = 100, range = None):
        
        self.preparePlot("Δt [s]", "Häufigkeit")
        HistY, edges, plot = plt.hist(self.data["delta"][takeFrom], bins, range = range)
        plt.show()
        plt.close()
        self.data["hist"][saveTo] = ((edges[1:] + edges[:-1]) / 2, HistY, edges)
        
        if(self.data["diagnose"]):
            print("histogram done")
    
    
    
    def fitHist(self, takeFrom, saveTo, gaussParams = None, areaMin = None, areaMax = None):
        
        print(np.argmax(self.data["hist"][takeFrom][1]))
        if(gaussParams is None):
            gaussParamsLoop = [1e-10, self.data["hist"][takeFrom][0][np.argmax(self.data["hist"][takeFrom][1])], 1e-12]
        else:
            gaussParamsLoop = gaussParams
        
        fit_params, var_matrix = curve_fit(self._gauss, self.data["hist"][takeFrom][0][areaMin:areaMax], self.data["hist"][takeFrom][1][areaMin:areaMax], p0 = gaussParamsLoop)
        self.preparePlot("Δt [s]", "Häufigkeit")
        plt.plot(self.data["hist"][takeFrom][0], self.data["hist"][takeFrom][1])
        plt.plot(self.data["hist"][takeFrom][0], self._gauss(self.data["hist"][takeFrom][0], fit_params))
        plt.show()
        print("parameter:", fit_params, "[A, mu, sigma]")
        
        self.data["result"][saveTo] = fit_params
        
        if(self.data["diagnose"]):
            print("fitHist done")
    
    
    
    def showResult(self, takeFrom):
    
        DoubleJitter = self.data["result"][takeFrom][2]
        print("σ_0:", DoubleJitter * 1e12, "ps")
    
        SingleJitter = DoubleJitter / math.sqrt(2)
        print("σ_1:", SingleJitter * 1e12, "ps")
        print("σ_1 = σ_0 / sqrt(2)")
        
        Mu = self.data["result"][takeFrom][1]
        print("\nμ:", Mu * 1e12, "ps")
        
        print("\ncoma separated [σ_0, σ_1, μ]")
        print("{},{},{}".format(DoubleJitter, SingleJitter, Mu))
        print("\n tab separated [σ_0, σ_1, μ]")
        print("{}\t{}\t{}".format(DoubleJitter, SingleJitter, Mu))
        
    
        
        
        
    def exportDict(self):
                        
        if(self.data["diagnose"]):
            print("exportDict done")

        return self.data
    
    
    
    
    def importDict(self, importDict):
        
        self.data = importDict
        
        if(self.data["diagnose"]):
            print("importDict done")
    
    
    

