# L2Boost.py

import numpy as np

class L2Boost(object):
    """L2-boost for high dimensional linear models.

    Parameters
    ----------
    inputMatrix: array
        nxp-Design matrix of the linear model.

    outputVarible: array
        n-dim vector of the observed data in the linear model.

    learningRate: float, default = 1
        Adjustment parameter for the size of the boosting steps.

    includeAic: bool, default = False
        If True, a generalised AIC criterion for model selection is computed
        alongside the boosting procedure.         

    trueSignal: array or None, default = None 
        For simulation purposes only. For simulated data the true signal can be
        included to compute theoretical quantities such as the bias and the mse
        alongside the boosting procedure.

    Attributes
    ----------
    sampleSize: int
        Sample size of the linear model
    
    paraSize: int
        Parameter size of the linear model

    iter: int
        Current boosting iteration of the algorithm

    trueSignal: array or None
        Only exists if parameter trueSignal was given.

    boostEstimate: array
        Boosting estimate at the current iteration for the data given in
        inputMatrix

    coefficients: array
        Estimated coefficients of the linear model at the current boosting
        iteration.

    componentDirections: array
        Lists the component directions the boosting procedure has chosen up to
        the current iteration.

    residuals: array
        Lists the sequence of the residual mean of squares betwean the data and
        the boosting estimator.

    bias2: array
        Only exists if trueSignal was given. Lists the values of the squared
        bias up to current boosting iteration.

    stochError: array
        Only exists if trueSignal was given. Lists the values of a stochastic
        error term up to current boosting iteration.

    mse: array
        Only exists if trueSignal was given. List the values of the mean squared
        error betwean the boosting estimator and the true signal up to current
        boosting iteration.
    """

    def __init__(self, inputMatrix, outputVariable, learningRate = 1, includeAic = False, trueSignal = None):
        self.inputMatrix    = inputMatrix
        self.outputVariable = outputVariable
        self.learningRate   = learningRate    
        self.trueSignal     = trueSignal
        self.__includeAic   = includeAic

        self.sampleSize = np.shape(inputMatrix)[0]
        self.paraSize   = np.shape(inputMatrix)[1]
        self.iter       = 0

        self.boostEstimate         = np.zeros(self.sampleSize)
        self.coefficients          = np.zeros(self.paraSize)
        self.componentDirections   = np.array([])

        self.__residualVector      = outputVariable
        self.residuals             = np.array([np.mean(self.__residualVector**2)])

        if self.__includeAic: 
            self.aic = np.array([np.log(self.residuals[self.iter]) +
                       1 / (1 - 2 / self.sampleSize)])
            self.aicClassic = np.array([self.sampleSize*np.log(np.mean(self.residuals[self.iter]**2)) +
                       2 * 0])
            self.__residualOperator = np.eye(self.sampleSize)

        if self.trueSignal is not None:
            self.__errorVector      = self.outputVariable - self.trueSignal 
            self.__bias2Vector      = self.trueSignal
            self.__stochErrorVector = np.zeros(self.sampleSize)

            self.bias2        = np.array([np.mean(self.__bias2Vector**2)])
            self.stochError   = np.array([0])
            self.stochErrorUB = np.array([0])
            self.mse          = np.array([np.mean(self.trueSignal**2)])

    def boost(self, m = 1):
        """Performs m steps of the boosting procedure"""
        for i in range(m): self.__boostOneIteration()

    def boostEarlyStop(self, crit, maxIter):
        """Early stopping for the boosting procedure

            Procedure is stopped when the residuals go below crit or iteration
            maxIter is reached.
        """
        while self.residuals[self.iter] > crit and self.iter <= maxIter:
            self.__boostOneIteration()

    def predict(self, inputVariable):
        """Predicts the output variable based on the current boosting estimate"""
        return np.dot(inputVariable, self.coefficients)

    def __boostOneIteration(self):
        weakLearner        = self.__computeWeakLearner()
        componentDirection = weakLearner.componentDirection
        stepSize           = weakLearner.stepSize

        self.boostEstimate    = self.boostEstimate    + stepSize * self.inputMatrix[:, componentDirection]
        self.__residualVector = self.__residualVector - stepSize * self.inputMatrix[:, componentDirection]

        self.residuals                         = np.append(self.residuals, np.mean(self.__residualVector**2))
        self.componentDirections               = np.append(self.componentDirections, componentDirection)
        self.iter                             += 1
        self.coefficients[componentDirection] += stepSize

        if self.__includeAic: self.__updateAic(componentDirection)

        if self.trueSignal is not None:
            self.__updateBias2(componentDirection)
            self.__updateStochError(componentDirection)
            self.mse = np.append(self.mse, np.mean((self.trueSignal - self.boostEstimate)**2))

    def __computeWeakLearner(self):
        stepSizes = np.zeros(self.paraSize)
        for j in range(self.paraSize):
            stepSizes[j] = np.dot(self.__residualVector, self.inputMatrix[:, j]) / np.sum(self.inputMatrix[:, j]**2)

        decreasedResiduals = np.zeros(self.paraSize)
        for j in range(self.paraSize):
            decreasedResiduals[j] = np.sum((self.__residualVector - stepSizes[j] * self.inputMatrix[:, j])**2)

        componentDirection = np.argmin(decreasedResiduals)
        correctedStepSize  = self.learningRate * stepSizes[componentDirection]
        return _WeakLearner(componentDirection, correctedStepSize)

    def __updateAic(self, componentDirection):
        direction               = self.inputMatrix[:, componentDirection]
        weakLearnerMatrix       = np.outer(direction, direction) / np.sum(direction**2)
        operatorUpdate          = (np.eye(self.sampleSize) - self.learningRate * weakLearnerMatrix)
        self.__residualOperator = operatorUpdate @ self.__residualOperator 

        degOfFreedom     = self.sampleSize - np.trace(self.__residualOperator) 
        currentResiduals = self.residuals[self.iter]
        newAic           = self.__computeAic(currentResiduals, degOfFreedom, self.sampleSize)
        self.aic         = np.append(self.aic, newAic)
        
    def __updateAicClassic(self, componentDirection):
        degOfFreedom     = self.sampleSize - np.trace(self.__residualOperator) 
        currentResiduals = self.residuals[self.iter]
        newAicClassic    = self.sampleSize * np.log(1/self.sampleSize * np.sum(currentResiduals**2)) + 2 * degOfFreedom
        self.aicClassic  = np.append(self.aicClassic, newAicClassic)

    def __updateBias2(self, componentDirection):
        direction          = self.inputMatrix[:, componentDirection]
        stepSize           = np.dot(self.__bias2Vector, direction) / np.sum(direction**2)
        correctedStepSize  = self.learningRate * stepSize
        self.__bias2Vector = self.__bias2Vector - correctedStepSize * direction
        self.bias2         = np.append(self.bias2, np.mean(self.__bias2Vector**2))

    def __updateStochError(self, componentDirection):
        direction              = self.inputMatrix[:, componentDirection]
        residualErrorVector    = self.__errorVector - self.__stochErrorVector
        stepSize               = np.dot(residualErrorVector, direction) / np.sum(direction**2)
        correctedStepSize      = self.learningRate * stepSize
        newResidualErrorVector = residualErrorVector - correctedStepSize * direction

        self.__stochErrorVector = self.__errorVector - newResidualErrorVector
        auxTerm1                = (2 * np.dot(self.__stochErrorVector, self.__errorVector) /
                                  self.sampleSize)
        auxTerm2                = np.mean(self.__stochErrorVector**2)
        self.stochError        = np.append(self.stochError, auxTerm2)
        self.stochErrorUB         = np.append(self.stochErrorUB, auxTerm1 - auxTerm2)

    def __computeAic(self, residuals, degOfFreedom, sampleSize):
        return (np.log(residuals) +
                (1 + degOfFreedom / sampleSize) / (1 - (degOfFreedom + 2) / sampleSize))


class _WeakLearner(object):
    """Weak learner consisting of component direction and step size"""

    def __init__(self, componentDirection, stepSize):
        self.componentDirection = componentDirection
        self.stepSize = stepSize