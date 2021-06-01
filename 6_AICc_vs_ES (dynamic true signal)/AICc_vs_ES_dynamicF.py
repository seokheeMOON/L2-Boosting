#!/usr/bin/env python
# coding: utf-8


#%%

### 0. Import relevant packages
from L2Boost import L2Boost
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import diags  # to generate correlated covariance matrices


#%%
### 1. Define the basic setups

# data setups
N = np.array([20,50,100,150,200,250,300])  # number of observations
P = 10 * 2**np.arange(len(N))  # dimensionality of the predictor space
s = 2  # error standard deviation
corr = False  # correlation between the predictor variables

# simulation setup
simNum = 1000  # number of simulations
maxBIter = 1500  # number of maximum allowed boosting iterations
avgSimData = pd.DataFrame({  # data frame to save the summary of the simulation
                             # for each setup (n,p) in zip(N,P): 
                             # average & standard deviation of MSE and stopping time
                            'stopTime_ES' : np.zeros(len(N)), 'mse_ES' : np.zeros(len(N)),
                            'stopTime_AICc' : np.zeros(len(N)), 'mse_AICc' : np.zeros(len(N)), 
                            'stopTime_min' : np.zeros(len(N)), 'mse_min' : np.zeros(len(N)),
                            'SDstopTime_ES' : np.zeros(len(N)), 'SDmse_ES' : np.zeros(len(N)),
                            'SDstopTime_AICc' : np.zeros(len(N)), 'SDmse_AICc' : np.zeros(len(N)), 
                            'SDstopTime_min' : np.zeros(len(N)), 'SDmse_min' : np.zeros(len(N))})


#%%
### 2. Simulation

for setupIdx, (n, p) in enumerate(zip(N,P)):
    
    # data frame to save the simulation results for a given setup (n,p)
    simData =pd.DataFrame({'stopTime_ES' : np.zeros(simNum), 'mse_ES' : np.zeros(simNum),
                          'stopTime_AICc' : np.zeros(simNum), 'mse_AICc' : np.zeros(simNum),
                          'stopTime_min' : np.zeros(simNum), 'mse_min' : np.zeros(simNum)})

    # Define a (linearly growing) true signal function
    def f(X): 

        c = 19
        coef = np.array([round(1/(k**2),3) for k in range(1,3*(setIdx+1)+1)])
        
        return 1 + c * (np.sum(coef * X[:, 1:(3*(setIdx+1)+1)], 1))

    # start simulations
    for iterSim in range(simNum):
        
        
        ## 2-1. Generate sample data

        # X: Predictor variables
        X = np.random.multivariate_normal(mean = np.zeros(p), cov = np.eye(p), size = n)
        X = np.hstack((np.ones(n)[:,None], X))  # add an offset variable 
                
        # eps: Error term
        eps = np.random.normal(loc = 0, scale = s, size = n)
        
        # Y: Observed variable
        Y = f(X) + eps
        
        
        ## 2-2. Run L2-Boosting
        l2 = L2Boost(inputMatrix=X, outputVariable=Y, learningRate = 0.1, includeAic = True, trueSignal = f(X))
        l2.boost(m=maxBIter)


        ## 2-3. Save the simulation result

        # Early stopping
        critES = np.argwhere(l2.residuals <= s**2)  # Early stopping criterion: 
                                                    # stop if the average residual becomes smaller than equal to the error variance (s**2)
        if critES.any():    # The criterion is met during boosting iterations m = 1,...,maxBIter
            stopTime_ES = critES[0]
        else:   # The criterion is not met during the allowed boosting iterations
            stopTime_ES = maxBIter

        simData.stopTime_ES[iterSim] = stopTime_ES
        simData.mse_ES[iterSim] = l2.mse[stopTime_ES]

        # Corrected AIC
        stopTime_AICc = l2.aic.argmin()

        simData.stopTime_AICc[iterSim] = stopTime_AICc
        simData.mse_AICc[iterSim] = l2.mse[stopTime_AICc]

        # Min MSE
        simData.mse_min[iterSim] = l2.mse.min()
        simData.stopTime_min[iterSim] = l2.mse.argmin()

        
    # save average statistics of the simulations
    avgSimData.iloc[setIdx,:6] = simData.mean(0)  # average of the estimates
    SD = simData.std(0)  # standard deviation of the estimates
    SD.index = ['SDstopTime_ES', 'SDmse_ES','SDstopTime_AICc', 'SDmse_AICc','SDstopTime_min', 'SDmse_min']
    avgSimData.iloc[setIdx,6:] = SD


#%%

### 3. Plot the simulation results (Summary statistics)

fig, axes = plt.subplots(2, 1, figsize = (10,14))
plt.subplots_adjust(hspace=0.25)
ax1, ax2 = axes

## 3-1. MSE
ax1.plot(avgSimData.mse_AICc, '.-r', label = 'AICc')
ax1.plot(avgSimData.mse_ES, '.-b', label = 'ES')
ax1.plot(avgSimData.mse_min, '.-k', label = 'Min')
ax1.grid(axis = 'y', linestyle = '--')
ax1.set_xticks(range(setupIdx+1))
ax1.set_xticklabels([f'({n},{p})' for n,p in zip(N,P)], fontsize=12)
ax1.set_title('MSE', fontsize=18)
ax1.set_xlabel('Settings: (n,p)', fontsize = 15)
_=ax1.legend(frameon=False, fontsize = 12)

## 3-2. Stopping time
ax2.plot(avgSimData.stopTime_AICc, '.-r', label = 'AICc')
ax2.plot(avgSimData.stopTime_ES, '.-b', label = 'ES')
ax2.plot(avgSimData.stopTime_min, '.-k', label = 'Min')
ax2.grid(axis = 'y', linestyle = '--')
ax2.set_xticks(range(setupIdx+1))
ax2.set_xticklabels([f'({n},{p})' for n,p in zip(N,P)], fontsize=12)
ax2.set_title('Stopping Time', fontsize=18)
ax2.set_xlabel('Settings: (n,p)', fontsize = 15)
_=ax2.legend(frameon=False, fontsize = 12)

