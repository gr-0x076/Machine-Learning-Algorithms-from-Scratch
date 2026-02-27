import numpy as np

class RidgeRegressionScratch:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        
        X_b = np.c_[np.ones((n_samples,1)), X]
        
        I = np.eye(n_features + 1)
        I[0,0] = 0 
        
        self.weights = np.linalg.inv(
            X_b.T @ X_b + self.alpha * I
        ) @ X_b.T @ y
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples,1)), X]
        return X_b @ self.weights