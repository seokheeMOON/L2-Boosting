#!/usr/bin/env python
# coding: utf-8

#%%
import os

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
n = 100  # number of observations
p = 40  # dimensionality of the predictor space
s = 2  # error standard deviation
corr = False  # correlation between the predictor variables

# simulation setup
simNum = 2  # number of simulations
maxBIter = 600  # number of maximum allowed boosting iterations


### 2. Simulation 

# matrices to save the simulation results
mse = np.empty((simNum, maxBIter+1))
stochErr = np.empty((simNum, maxBIter+1))
bias2 = np.empty((simNum, maxBIter+1))
ES = np.empty(simNum)

 # start simulations
for iterSim in range(simNum):
    
    ## 2-1. Generate sample data

    # X: Predictor variables
    if corr == False:   # uncorrelated X
        a = 1   # scaling factor
        X = np.random.multivariate_normal(mean = np.zeros(p), cov = np.eye(p), size = n)
    else:  # correlated X 
        a = 0.779  # scaling factor
        b = 0.677  # correlation in the second diagonals
        c = 0.323  # correlation in the third diagonals
        
        # box correlation matrix
        cov = np.array([c*np.ones(p-2), b*np.ones(p-1),np.ones(p),b*np.ones(p-1), c*np.ones(p-2)])
        offset = [-2,-1,0,1,2]
        cov = diags(cov,offset).toarray()
        
        X = np.random.multivariate_normal(mean = np.zeros(p), cov = cov, size = n)

    X = np.hstack((np.ones(n)[:,None], X))  # add an offset variable 
    
    # f: True underlying relationship btw Y and X
    def f(X): return a*(1 + 5 * X[:,1] + 2 * X[:,2] + X[:,3])
    
    # eps: Error term
    eps = np.random.normal(loc = 0, scale = s, size = n)
    
    # Y: Observed variable
    Y = f(X) + eps
    
    
    ## 2-2. Run L2-Boosting
    l2 = L2Boost(inputMatrix=X, outputVariable=Y, learningRate = 0.1, includeAic = True, trueSignal = f(X))
    l2.boost(m=maxBIter)


    ## 2-3. Save the simulation result
    mse[iterSim,:] = l2.mse
    bias2[iterSim,:] = l2.bias2
    stochErr[iterSim,:] = l2.stochError
    
    # Early stopping
    critES = np.argwhere(l2.residuals <= s**2)  # Early stopping criterion: 
                                                # stop if the average residual becomes smaller than equal to the error variance (s**2)
    if critES.any():    # The criterion is met during boosting iterations m = 1,...,maxBIter
        stopTime_ES = critES[0]
    else:   # The criterion is not met during the allowed boosting iterations
        stopTime_ES = maxBIter

    ES[iterSim] = stopTime_ES



#%%

### 3. Plot the simulation results (Summary statistics)

fig, ax = plt.subplots(figsize = (10,7))

## 3-1. MSE
ax.plot(mse.mean(0), '-k', label = 'MSE')
ax.plot(bias2.mean(0), '-b', label = 'Squared bias')
ax.plot(stochErr.mean(0), '-r', label = 'Variance')
ax.vlines(ES.mean(),0,3, color = 'grey', linestyle = '--')
ax.set_ylim(0,3)
ax.grid(axis = 'y', linestyle = '--')
ax.set_title('Bias-variance Trade-off', fontsize = 18)
ax.set_xlabel('# boosting iterations', fontsize = 15)
_=ax.legend(frameon=False, fontsize = 12)

