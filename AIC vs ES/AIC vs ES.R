# 1 Load packages --------------------------------------------------------------
library(mboost)
library(MASS)  # for mvrnorm function
library(gifski)  # for making a gif from the graphical results


# 2 Basic Set-ups -------------------------------------------------

# Set Parameters
simNum = 1000  # number of simulations 
maxBIter = 600  # number of the maximum boosting iterations 

n = 20  # number of observations
p = 10  # number of the predictor variables
s = 2   # error sd

# Generate a data frame to save the simulation results
simData <- data.frame('stopTime_ES' = double(simNum), 'mse_ES' = double(simNum),
                      'stopTime_AIC' = double(simNum), 'mse_AIC' = double(simNum), 
                      'mse_min' = double(simNum))


# 3 Simulation -----------------------------------------------------------------
for (iterSim in 1:simNum) {
  
  # * 3.1 Generate the data ----------------------------------------------------
  
  # Predictor variables
  X = mvrnorm(n = n, mu = rep(0,p), Sigma = diag(p))
  a = 1  # scaling factor
  
  # True underlying relationship btw Y and X
  f = a * (1 + 5*X[,1] + 2*X[,2] + X[,3])
  
  # eps
  eps = rnorm(n = n, mean = 0, sd = s)
  
  # Dependent variable
  Y = f + eps
  
  # Named data frame for training
  dfTrain = data.frame('Y' = Y, 'X' = X)  # glmboost needs a df as an input.
  
  
  # * 3.2 L2-Boosting ----------------------------------------------------------
  
  # Train the models
  L2 = glmboost(Y ~ ., data = dfTrain)

  # Generate empty vectors to save Residual & MSEs at each boosting iteration
  resVec <- rep(0, maxBIter)
  mseVec <- rep(0, maxBIter)  
  
  # Calculate the Residual & MSE for each 
  for (iterBoost in 1:maxBIter) {
    
    l2 = L2[iterBoost]  # model at iterBoost-th iteration
    
    # L2 Boosting fit 
    Fhat = predict(l2, newdata = dfTrain[-1])  
    
    # Residual mean squared
    resVec[iterBoost] = mean((dfTrain$Y - Fhat)^2)  
    
    # MSE
    mseVec[iterBoost] = mean((f - Fhat)^2) 
    
  }
  
  # * 3.3 Early stoping (In-Sample) --------------------------------------------
  
  # Mark 1 if the residual mean squared becomes smaller than or equal to the error variance
  idx_ES = ifelse(resVec <= s^2, 1, 0)
  
  # Stopping time of Early Stopping
  stopTime_ES = ifelse(sum(idx_ES)>=1, which(idx_ES == 1)[1], maxBIter)
  
  # Save the stopping time & MSE for Early Stopping
  simData$stopTime_ES[iterSim] = stopTime_ES
  simData$mse_ES[iterSim] = mseVec[stopTime_ES]
  
  
  # * 3.4 Corrected AIC --------------------------------------------------------
  
  # Stopping time of corrected AIC
  stopTime_AIC = mstop(AIC(L2, method = 'corrected'))
  
  # Save the stopping time & MSE for corrected AIC
  simData$stopTime_AIC[iterSim] = stopTime_AIC
  simData$mse_AIC[iterSim] = mseVec[stopTime_AIC]
  
  # * 3.5 Minimun MSE ----------------------------------------------------------
  simData$mse_min[iterSim] = min(mseVec)
  
  
  # * 3.6 Plot the simulation result -------------------------------------------
  # setwd('C:/Users/Seokhee2/Desktop/graphics')
  # file.name = paste0('n',n,'p',p,'iterSim',iterSim,'maxBIter',maxBIter,'.png')
  # png(file.name,width = 1200, height = 834, pointsize = 25)
  plot(mseVec, type = 'l', col = 'black', xlab = '# of iteration',
       ylab = paste0('MSE at sim # ', iterSim), ylim = c(0,30))
  abline(v = stopTime_AIC, col = 'red')
  abline(h = mseVec[stopTime_AIC], col = 'red', lty='dotted')
  abline(v = stopTime_ES, col = 'blue')
  abline(h = mseVec[stopTime_ES], col = 'blue', lty='dotted')
  abline(h = min(mseVec), col = 'black', lty = 'dotdash')
  abline(h = s^2, col = 'grey', lty = 'dotted')
  legend("topright",
         c("L2 Boosting / min MSE", "corrected AIC", 'in-sample early stopping', 'error variance'),
         fill = c("black", "red", 'blue', 'grey'))
  # dev.off()
    

}

# 4 Export the results ---------------------------------------------------------

# Numerical results
# write.csv(simData, 'AIC_vs_ES.csv')

# Graphical results
# png_files <- list.files('C:/Users/Seokhee2/Desktop/graphics', pattern = ".*png$", full.names = TRUE)
# gifski(png_files, gif_file = "AIC_vs_ES.gif", width = 800, height = 600, delay = 2)
