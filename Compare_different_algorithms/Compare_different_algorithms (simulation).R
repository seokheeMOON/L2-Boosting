###############################################################################
###################  Comparison over different Methods  #######################
###############################################################################

library(MASS)
library(glmnet)  # try out l2boost
library(mboost)

# define a performance measure
MSE <- function(y, yhat){
  
  mean((y - yhat)^2)
  
}


M = 10  # number of simulations
B = 200  # number of maximum boosting iterations
corr = FALSE  # decide whether to use an uncorrelated X or not

# create vectors to store the results of every simulation
MSE_ols = c()
MSE_fwdVS = c()
MSE_ridge = c()
MSE_lasso = c()
MSE_l2 = matrix(nrow = M, ncol = B)
MSE_l2_minAIC = c()

for (m in 1:M) {
  
  ########################################
  ## generate the data
  ########################################
  
  n = 10  # number of observations
  p = 30   # dimension
  
  
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
  epsTrain = rnorm(n = n, mean = 0, sd = 2)
  Ytrain = f(Xtrain) + epsTrain
  
  # named data frame for the train data
  Dtrain = data.frame('Y' = Ytrain, 'X' = Xtrain)
  
  ## generate the test data ##
  
  if (corr == FALSE){
    
    # X variables without correlation
    Xtest = mvrnorm(n = n, mu = rep(0,p), Sigma = diag(p))

  } else {
    
    # X variables with block correlation
    Xtest = mvrnorm(n = n, mu = rep(0,p), Sigma = Sig)
  }
  
  # generate eps and Y
  epsTest = rnorm(n = n, mean = 0, sd = 2)
  Ytest = f(Xtest) + epsTest
  
  # named data frame for the test data
  Dtest = data.frame('Y' = Ytest, 'X' = Xtest)
  
  ########################################
  ## OLS
  ########################################
  
  # train the model
  ols = lm(Y ~ ., data = Dtrain)
  
  b_ols = ols$coefficients
  
  # make predictions for the new data
  y_ols = predict(ols, newdata = Dtest[-1])
  
  # save the performance score
  MSE_ols[m] = MSE(Ytest, y_ols)

  
  ########################################
  ## Forward Variable Selection
  ########################################
  
  # train the model 
  full.model = lm(Y ~ ., data = Dtrain)
  offset.model = lm(Y ~ 1, data = Dtrain)  # offset model only with a constant regressor
  fwdVS = stepAIC(offset.model, direction="forward",
                  scope = formula(full.model), trace = 0)  # model selection + training  
  
  b_fwdVS = fwdVS$coefficients
  
  # make predictions
  y_fwdVS = predict(fwdVS, newdata = Dtest[-1])
  
  # save the performance score
  MSE_fwdVS[m] = MSE(Ytest, y_fwdVS)

  
  ########################################
  ## Ridge
  ########################################

  # named matrix/vector version of the variables (needed for glmnet)
  xtrain = as.matrix(Dtrain[-1])
  ytrain = as.matrix(Dtrain[1])
  
  xtest = as.matrix(Dtest[-1])

  # CV for hyper parameter selection
  cv_ridge <- cv.glmnet(x = xtrain, y = ytrain,
                        alpha = 0)  # alpha=0 for Ridge regression
  lam_opt <- cv_ridge$lambda.min  # or $lambda.1se
  
  # train the model 
  ridge = glmnet(x = xtrain, y = ytrain, 
                 family = 'gaussian', alpha = 0, lambda = lam_opt)   # alpha=0 for Ridge regression
     
  b_ridge = predict(ridge, type='coefficients') 
  
  # make predictions
  y_ridge = predict(ridge, newx = xtest)

  # save the performance score
  MSE_ridge[m] = MSE(Ytest, y_ridge)
  

  ########################################
  ## LASSO
  ########################################
  
  # CV for hyper parameter selection
  cv_lasso <- cv.glmnet(x = xtrain, y = ytrain, alpha = 1)  # alpha=1 for LASSO regression
  lam_opt <- cv_lasso$lambda.min  # or $lambda.1se
  
  # train the model 
  lasso = glmnet(x = xtrain, y = ytrain, family = 'gaussian', alpha = 1, lambda = lam_opt)   # alpha=1 for LASSO regression
  
  b_lasso = predict(lasso, type='coefficients') 

    # make predictions
  y_lasso = predict(lasso, newx = xtest)
  
  # save the performance score
  MSE_lasso[m] = MSE(Ytest,y_lasso)
  
  
  ########################################
  ## L2 Boosting
  ########################################
  
  # train the model
  L2 = glmboost(Y ~ ., data = Dtrain)
  for (b in 1:B) {
    l2 = L2[b]  # model at iteration b
    y_l2 = predict(l2, newdata = Dtest[-1])  # predictions at iteration b
    MSE_l2[m,b] = MSE(Ytest, y_l2)  # save the performance score
  }
  
  # find a model with the min AIC value and save its performance score
  minAIC = mstop(aic <- AIC(l2))
  MSE_l2_minAIC[m] = MSE_l2[m, minAIC]
  
}


# plot the average results
plot(colMeans(MSE_l2), type = 'l', ylim = c(0,40), xlab = '# of iterations',
     ylab = 'Mean MSE')
abline(h = mean(MSE_lasso), col = 'red')
abline(h = mean(MSE_ols), col = 'blue')
abline(h = mean(MSE_fwdVS), col = 'green')
abline(h = mean(MSE_ridge), col = 'purple')
abline(h = mean(MSE_l2_minAIC), col = 'black', lty = 'dotted')
legend("topright",
       c("L2", "lasso", 'ols','fwdVs', 'ridge'),
       fill = c("black", "red", 'blue','green', 'purple'))


