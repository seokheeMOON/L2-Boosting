###############################################################################
############### Comparison over different Stopping Criteria  ##################
###############################################################################
rm(list = ls())

library(MASS)
library(glmnet)
library(mboost)

# define a performance measure
MSE <- function(y, yhat){
  
  mean((y - yhat)^2)
  
}


M = 5  # number of simulations
B = 600  # number of maximum boosting iterations
corr = FALSE  # decide whether to use an uncorrelated X or not

# to store performance of the models at every iter
MSEtrain = matrix(nrow = M, ncol = B)
MSEtest = matrix(nrow = M, ncol = B)
MSEval = matrix(nrow = M, ncol = B)

# to store optimal stopping point for different criteria
mstop = data.frame('AIC' = double(M), 'gMDL' = double(M), 'CV10f' = double(M), 
               'CVbs25' = double(M), 'estp_oosp' = double(M), 'estp_insp' = double(M))

# to store performance of each optimal models
Acc = data.frame('AIC' = double(M), 'gMDL' = double(M), 'CV10f' = double(M), 
                      'CVbs25' = double(M), 'estp_oosp' = double(M), 'estp_insp' = double(M))
Corr = data.frame('AIC' = double(M), 'gMDL' = double(M), 'CV10f' = double(M), 
                  'CVbs25' = double(M), 'estp_oosp' = double(M), 'estp_insp' = double(M))

for (m in 1:M) {
  
  ########################################
  ## generate the data
  ########################################
  
  n = 100  # number of observations
  p = 30   # dimension
  s = 2  # error sd
  
  
  ## Generate the train data ##
  
  # X variables
  if (corr == FALSE){
    
    # X without correlation
    Xtrain = mvrnorm(n = n, mu = rep(0,p), Sigma = diag(p))
    a = 1  # scaling factor
  } else {
    
    # define a block covariance matrix of X
    a = 0.779  # scaling factor
    b = 0.677  # correlation in the second diagonals
    c = 0.323  # correlation in the third diagonals
    Sig = diag(p)
    diag(Sig[-1,]) = b
    diag(Sig[,-1]) = b
    diag(Sig[-c(1,2),]) = c
    diag(Sig[,-c(1,2)]) = c
    
    # X  with block correlation
    Xtrain = mvrnorm(n = n, mu = rep(0,p), Sigma = Sig)
  }
  
  # define the underlying relationship f btw Y and X
  f <- function(X) {
    a * (1 + 5*X[,1] + 2*X[,2] + X[,3])
  }
  
  # eps and Y
  epsTrain = rnorm(n = n, mean = 0, sd = s)
  Ytrain = f(Xtrain) + epsTrain
  
  # named data frame for the train data
  Dtrain = data.frame('Y' = Ytrain, 'X' = Xtrain)
  
  
  ## generate the test data (for model selection) ##
  
  if (corr == FALSE){
    
    # X variables without correlation
    Xtest = mvrnorm(n = n, mu = rep(0,p), Sigma = diag(p))
    
  } else {
    
    # X variables with block correlation
    Xtest = mvrnorm(n = n, mu = rep(0,p), Sigma = Sig)
  }
  
  # generate eps and Y
  epsTest = rnorm(n = n, mean = 0, sd = s)
  Ytest = f(Xtest) + epsTest
  
  # named data frame for the test data
  Dtest = data.frame('Y' = Ytest, 'X' = Xtest)
  
  
  ## generate the validation data (for prediction) ##
  
  if (corr == FALSE){
    
    # X variables without correlation
    Xval = mvrnorm(n = n, mu = rep(0,p), Sigma = diag(p))
    
  } else {
    
    # X variables with block correlation
    Xval = mvrnorm(n = n, mu = rep(0,p), Sigma = Sig)
  }
  
  # generate eps and Y
  epsVal = rnorm(n = n, mean = 0, sd = s)
  Yval = f(Xval) + epsVal
  
  # named data frame for the test data
  Dval = data.frame('Y' = Yval, 'X' = Xval)
  
  ########################################
  ## L2 Boosting
  ########################################
  
  # train the model
  # options(mboost_dftraceS = TRUE) 
  L2 = glmboost(Y ~ ., data = Dtrain)
  estp_oosp = c()  # save 1 if out of sample MSE decreases compared to the previous iter, and 0 otherwise
  estp_insp = c()  # save 1 if in sample MSE becomes smaller than a pre-set threshold, e.g. error variance
  for (b in 1:B) {
    l2 = L2[b]  # model at iteration b
    y_fit = predict(l2, newdata = Dtrain[-1])  # fit at iter b
    MSEtrain[m,b] = MSE(Dtrain$Y, y_fit)  # MSE on the training set
    estp_insp[b] = ifelse(MSEtrain[m,b] <= s^2, 1, 0)
    
    y_test = predict(l2, newdata = Dtest[-1])  # predict at iter b on the test set (for early stopping)
    MSEtest[m,b] = MSE(Dtest$Y, y_test)
    estp_oosp[b] = ifelse(MSEtest[m,b] > MSEtest[m,b-1], 1, 0)
    
    y_val = predict(l2, newdata = Dval[-1])  # predict at iter b on the validation set
    MSEval[m,b] = MSE(Dval$Y, y_val)
    
  }
  
  # out of sample early stopping
  idx_oosp = which(estp_oosp == 1)[1]  # Other rules can be applied as well, e.g. stop when the out of sample error decreases 5 times in a row.
  mstop$estp_oosp[m] = idx_oosp - 1  
  
  
  # in sample early stopping
  idx_insp = which(estp_insp == 1)[1]
  mstop$estp_insp[m] = idx_insp - 1 
  
  
  # corrected AIC 
  mstop$AIC[m] = mstop(AIC(L2, method = 'corrected'))
  
  # gMDL
  mstop$gMDL[m] = mstop(AIC(L2, method = 'gMDL'))
  
  # 10-fold cross-validation
  cv10f = cv(model.weights(L2), type = "kfold", B = 10)
  mstop$CV10f[m] = mstop(cvrisk(L2, folds = cv10f, papply = lapply))
  
  # 25 bootstrap iterations (manually)
  bs25 = cv(model.weights(L2), type = "bootstrap", B = 25)
  mstop$CVbs25[m] = mstop(cvrisk(L2, folds = bs25, papply = lapply))
  
  
  ########################################
  ## Validation
  ########################################
  
  for (i in 1:ncol(Acc)) {
    
    model_idx = mstop[m,i]
    model = L2[model_idx]
    y_val = predict(model, newdata = Dval[-1])
    
    # Accuracy/Correlation of each stopping criteria
    Acc[m, i] = MSEval[m, model_idx]
    Corr[m, i] = cor(Dval$Y, y_val)
    
  }
    

    
}


setwd('C:/Users/Seokhee2/Desktop/graphics')
file.name = paste0('n',n,'p',p,'M',M,'B',B,'corr',corr,'.png')
png(file.name,width = 1200, height = 834, pointsize = 25,bg = "transparent")
plot(colMeans(MSEval), type = 'l', ylim = c(0,20), xlab = '# of iteration',
     ylab = 'Mean MSE')
abline(v = mean(mstop$AIC), col = 'red')
abline(h = mean(Acc$AIC), col = 'red', lty='dotted')
abline(v = mean(mstop$gMDL), col = 'blue')
abline(h = mean(Acc$gMDL), col = 'blue', lty='dotted')
abline(v = mean(mstop$CV10f), col = 'green')
abline(h = mean(Acc$CV10f), col = 'green', lty='dotted')
abline(v = mean(mstop$CVbs25), col = 'purple')
abline(h = mean(Acc$CVbs25), col = 'purple', lty='dotted')
abline(v = mean(mstop$estp_oosp), col = 'orange')
abline(h = mean(Acc$estp_oosp), col = 'orange', lty='dotted')
abline(v = mean(mstop$estp_insp), col = 'pink')
abline(h = mean(Acc$estp_insp), col = 'pink', lty='dotted')
legend("topright",
       c("L2 Boosting", "corrected AIC", 'gMDL','CV 10 fold', 'CV bootstrap', 
         'out-of-sample early stopping', 'in-sample early stopping'),
       fill = c("black", "red", 'blue','green', 'purple', 'orange', 'pink'))
dev.off()

colMeans(Acc)
colMeans(Corr)

colMeans(MSEtrain)[599]
