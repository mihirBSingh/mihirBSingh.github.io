import torch

class LinearModel:

    def __init__(self):
        self.w = None 
        self.prevWeight = None
        
    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        return X@(self.w)

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        y_hat = (scores > 0.0) * 1.0
        return y_hat

class LogisticRegression(LinearModel):
        
    def loss(self, X, y):
        """
        Compute the empirical risk L(w) using the logistic loss function

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)

        RETURNS: 
            torch.Tensor: float: the loss L(w)
        """
        
        # s is score
        s = self.score(X)
        
        sigma = 1.0/(1.0+torch.exp(-s))
        sigma[sigma == 1.0] = 0.9999999
        sigma[sigma == 0.0] = 0.0000001
        
        return torch.mean(-y*torch.log(sigma) - (1-y)*torch.log(1-sigma))
    
    def grad(self, X, y):
        """
        Computes the gradient of the empirical risk L(w).

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)

        RETURNS: 
            torch.Tensor: float: the empirical risk gradient
        """
        
        s = self.score(X)
        
        sigma = 1/(1+torch.exp(-s))
        sigma[sigma == 1.0] = 0.9999999
        sigma[sigma == 0.0] = 0.0000001
        
        return torch.mean((sigma-y)[:,None]*X, dim=0)
    
    def hessian(self, X, y):
        """
        Computes the Hessian of the empirical risk L(w).

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels. y.size() = (n,)

        RETURNS: 
            torch.Tensor: float: the empirical risk Hessian
        """
        
        s = self.score(X)
        sigma = torch.sigmoid(s)
        
        return X.T@(torch.diag(sigma * (1-sigma)))@X
            
class GradientDescentOptimizer(LogisticRegression):
    
    def __init__(self, model):
        self.model = model
        
    def step(self, X, y, alpha, beta):
        """
        Implements gradient descent with momentum (spicy gradient descent) to compute the update of model's weight vector for a single step 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 
            
            y, torch.Tensor: the vector of target labels (0 or 1). y.size() = (n,)
            
            alpha, float: the learning rate
            beta, float: the momentum parameter

        RETURNS: 
            torch.Tensor: float: the loss L(w)
        """
        
        grad = self.model.grad(X, y)
        weight = self.model.w 
        
        if(self.model.prevWeight != None):
            self.model.w = weight - alpha*grad + beta*(weight - self.model.prevWeight)
            self.model.prevWeight = self.model.w
        else:  
            # make some random starting weight   
            self.model.prevWeight = torch.rand(X.size()[1])
                
        return self.model.loss(X, y)
    
class NewtonOptimizer(LogisticRegression):
    
    def __init__(self, model):
        self.model = model
        
    def step(self, X, y, alpha):
        
        # get gradient of L
        grad = self.model.grad(X, y)
        loss = self.model.loss(X, y)
                
        # get hessian of L
        hessian = self.model.hessian(X, y)
        
        # get new weight
        self.model.w = self.model.w - (alpha*torch.linalg.inv(hessian))@grad
        
        return loss