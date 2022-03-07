# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:16 2020

@author: joerg_j
"""


import matplotlib.pyplot as plt
import numpy as np
import os
from readTrc import Trc
import re
import math
from scipy.optimize import curve_fit


def realGaussian(x, A, mu = None, sigma = None):
    if(mu == None):
        A, mu, sigma = A
    return A * np.exp(-np.power((x - mu) / sigma, 2) / 2) / (sigma * math.sqrt(2 * math.pi))




def readChannel(channel, folderPath, traceName = "Trace", pulseMin = None, pulseMax = None):
    
    data = []
    meta = []
    for fn in os.listdir(folderPath):
        if(re.match(r"^" + channel + traceName + "[0-9]{5}\.trc$", fn)):
            if(pulseMin is not None):
                if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) < pulseMin:
                    continue
            if(pulseMax is not None):
                if int(re.match(r"^" + channel + traceName + "([0-9]{5})\.trc$", fn)[1]) >= pulseMax:
                    continue
            trc = Trc()
            trc.open(os.path.join(folderPath, fn))
            data.append(np.array((trc.x, trc.y)))
            meta.append(trc.d)
    return {
        "data": np.array(data),
        "meta": meta
    }
  



def paths(measurement_id, campaign_id, path_to_raw_data, path_to_hdf5_folder):

    import os
    
    return [os.path.join(path_to_raw_data, campaign_id, measurement_id), os.path.join(path_to_hdf5_folder, measurement_id + ".h5")]
    



def viewImport(C1, C3, pulsnr):
    
    plt.plot(C1["data"][pulsnr][0], C1["data"][pulsnr][1])
    plt.show()
    plt.plot(C3["data"][pulsnr][0], C3["data"][pulsnr][1])
    plt.show()
    
    plt.plot(C1["data"][pulsnr][0], C1["data"][pulsnr][1])
    plt.plot(C3["data"][pulsnr][0], C3["data"][pulsnr][1])
    plt.show()
    



def viewSelection(C, pulsnr, pulseMin, pulseMax):
    
    plt.plot(C["data"][pulsnr][0][pulseMin:pulseMax], C["data"][pulsnr][1][pulseMin:pulseMax], "-o")
    plt.show()




def fitChannel(data, areaMin, areaMax, gaussParams):
    fits = []

    for shot in data:
        
        try:
            fit_params, var_matrix = curve_fit(realGaussian, shot[0][areaMin:areaMax], shot[1][areaMin:areaMax], p0 = gaussParams)
        except:
            print("failed in shot", len(fits))
            return None
        
        fits.append(fit_params)
        
    return np.array(fits)




def minChannel(data):
    res = []
    for pulse in data["data"]:
        arg = np.argmin(pulse, axis=1)[1]
        time = pulse[0, arg]
        intensity = pulse[1, arg]
        res.append((arg, time, intensity))
    
    return np.array(res)
    



def threshold(C, factor, storage = "min", axis = 2, fallingEdge = True, show = None):
    
    arr = []
    
    for i in range(0, len(C["data"])):
        
        a = C["data"][i]

        yt = C[storage][i][axis] * factor
        
        if(fallingEdge):
            i2 = np.where(a[1] <= yt)[0][0]
            i1 = i2 - 1
        else:
            i1 = np.where(a[1] <= yt)[0][-1]
            i2 = i1 + 1
        try:
            x1 = a[0, i1]
            x2 = a[0, i2]
            y1 = a[1, i1]
            y2 = a[1, i2]
        except:
            print("failed in", i)
            return None
        
        xt = (x2 - x1) * (yt - y1) / (y2 - y1) + x1
        
        arr.append((xt, yt))
        
        if(show is not None):
            if(show == i):
                plt.plot(a[0], a[1])
                plt.plot(x1, y1, "o")
                plt.plot(x2, y2, "o")
                plt.plot([a[0, 0], a[0, -1]], [yt, yt], "-")
                plt.plot([xt, xt], [a[1, 0], C[storage][i][axis]], "-")
                plt.show()
                
                print("xt:", xt, "ps, yt", yt, "ps")
        
    return np.array(arr)



def viewFit(C, pulsnr, pulseMin, pulseMax, gaussParams):
    
    plt.plot(C["data"][pulsnr,0,:],C["data"][pulsnr,1,:])
    xFit = np.linspace(C["data"][pulsnr,0,pulseMin], C["data"][pulsnr,0,pulseMax], 50)
    plt.plot(xFit, realGaussian(xFit, C["fit"][pulsnr]))
    plt.show()
    print("parameter:", C["fit"][pulsnr], "[A, mu, sigma]")




def viewMu(C1, C3, pulseMin, pulseMax):
    
    plt.plot(C1["fit"][pulseMin:pulseMax,1])
    plt.plot(C3["fit"][pulseMin:pulseMax,1])
    plt.show()
    
    plt.plot(C1["fit"][pulseMin:pulseMax,1] - C3["fit"][pulseMin:pulseMax,1])
    plt.show()
    
    plt.plot(C1["fit"][:,1])
    plt.show()
    plt.plot(C3["fit"][:,1])
    plt.show()




def filterSpikes(C1, C3, max = True):
    
    print("function not yet implemented...")
    



def jitter(C1, C3, storage = "fit", axis = 1):
    
    return C1[storage][:,axis] - C3[storage][:,axis]




def histogram(JitterGauss, bins, range = None):
    
    HistY, edges, plot = plt.hist(JitterGauss, bins, range = range)
    return ((edges[1:] + edges[:-1]) / 2, HistY, edges)




def fitSingle(data, areaMin, areaMax, gaussParams):
        
    fit_params, var_matrix = curve_fit(realGaussian, data[0][areaMin:areaMax], data[1][areaMin:areaMax], p0 = gaussParams)
    plt.plot(data[0],data[1])
    plt.plot(data[0], realGaussian(data[0], fit_params))
    plt.show()
    print("parameter:", fit_params, "[A, mu, sigma]")
    return np.array(fit_params)




def result(HistFitParams):
    
    DoubleJitter = HistFitParams[2]
    print("σ_0:", DoubleJitter * 1e12, "ps")

    SingleJitter = DoubleJitter / math.sqrt(2)
    print("σ_1:", SingleJitter * 1e12, "ps")
    print("σ_1 = σ_0 / sqrt(2)")
    
    Mu = HistFitParams[1]
    print("\nμ:", Mu * 1e12, "ps")
    
    print("\ncoma separated [σ_0, σ_1, μ]")
    print("{},{},{}".format(DoubleJitter, SingleJitter, Mu))
    print("\n tab separated [σ_0, σ_1, μ]")
    print("{}\t{}\t{}".format(DoubleJitter, SingleJitter, Mu))
    
    return (DoubleJitter, SingleJitter, Mu)
    
    

