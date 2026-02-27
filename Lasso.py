class LassoScratch:
    
    def __init__(self, alpha=1.0, lr=0.01, n_iters=1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        X_b = np.c_[np.ones((n_samples,1)), X]
        self.weights = np.zeros(n_features + 1)
        
        for _ in range(self.n_iters):
            predictions = X_b @ self.weights
            errors = predictions - y
            
            gradient = (2/n_samples) * X_b.T @ errors
            
            
            l1_penalty = self.alpha * np.sign(self.weights)
            l1_penalty[0] = 0
            
            gradient += l1_penalty
            
            self.weights -= self.lr * gradient
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_b = np.c_[np.ones((n_samples,1)), X]
        return X_b @ self.weights
