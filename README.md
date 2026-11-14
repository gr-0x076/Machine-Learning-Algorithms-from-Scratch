# Machine Learning Algorithms from Scratch

NumPy implementations of foundational linear regression models built from scratch.

---

## Implemented Models

### 1. Ordinary Least Squares (`linearRegression.py`)
- **Optimization:** Batch gradient descent.
- **Model:** $\hat{y} = X_b w$, where $X_b = [\mathbf{1}, X]$ includes a prepended bias column.
- **Update Rule:**
  $$w \leftarrow w - \eta \cdot \frac{2}{n} X_b^T (X_b w - y)$$
- **Performance Metrics:** Implements `performance_metrics(X, y)` returning MSE, RMSE, MAE, $R^2$, and Adjusted $R^2$.

### 2. Ridge Regression (`Ridge.py`)
- **Optimization:** Closed-form analytical solution via regularized normal equations.
- **Analytical Solution:**
  $$w = (X_b^T X_b + \alpha I^*)^{-1} X_b^T y$$
- **Intercept Treatment:** The regularization identity matrix $I^*$ sets $I^*_{0,0} = 0$, explicitly excluding the bias term from $L_2$ regularization.

### 3. Lasso Regression (`Lasso.py`)
- **Optimization:** Subgradient descent.
- **Gradient Computation:**
  $$g = \frac{2}{n} X_b^T (X_b w - y) + \alpha \cdot p$$
  where $p = \text{sign}(w)$ with $p_0 = 0$ (the bias term is left unregularized).
- **Update Rule:**
  $$w \leftarrow w - \eta \cdot g$$
- **Implementation Note:** Uses naive subgradient descent with a fixed learning rate $\eta$, rather than coordinate descent with soft-thresholding.

---

## Usage Example

```python
import numpy as np
from Ridge import RidgeRegressionScratch

X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0]])
y = np.array([2.5, 3.8, 5.2])

model = RidgeRegressionScratch(alpha=0.5)
model.fit(X, y)
predictions = model.predict(X)
```

---

## Dependencies

- Python 3.8+
- NumPy