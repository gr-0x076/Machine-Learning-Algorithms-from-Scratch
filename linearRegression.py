import numpy as np
class LinearRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        x_b = np.c_[np.ones((n_samples, 1)), X]
        self.weights = np.zeros(n_features + 1)
        for _ in range(self.n_iters):
            predictions = x_b@self.weights
            errors = predictions - y
            gradient = (2/n_samples)*x_b.T@errors
            self.weights -= self.lr*gradient

    def predict(self, X):
        n_samples = X.shape[0]
        x_b = np.c_[np.ones((n_samples, 1)), X]
        return x_b@self.weights



    def performance_metrics(self, X, y):
        y_pred = self.predict(X)
        n = X.shape[0]
        p = X.shape[1]
        mse = np.mean((y - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        ssr = np.sum((y - y_pred)**2)
        sst = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ssr / sst)

        adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Adjusted_R2": adjusted_r2
        }
