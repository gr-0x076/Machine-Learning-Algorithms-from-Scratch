# Machine Learning Algorithms from Scratch

NumPy implementations of foundational linear models built from first principles without scikit-learn.

---

## Implemented Models

1. **Ordinary Least Squares (`linearRegression.py`):**
   - Optimization: Batch gradient descent.
   - Cost Function: Mean Squared Error (MSE).
   - Evaluation metrics: MSE, RMSE, R-squared ($R^2$), and Adjusted $R^2$.

2. **Ridge Regression (`Ridge.py`):**
   - Optimization: Closed-form analytical solution via the normal equations with $L_2$ regularization:
     $$w = (X^T X + \alpha I)^{-1} X^T y$$
   - Zero-penalizes the intercept term ($I_{0,0} = 0$).

3. **Lasso Regression (`Lasso.py`):**
   - Optimization: Subgradient descent with $L_1$ regularization penalty:
     $$\nabla J(w) = \frac{2}{n} X^T (Xw - y) + \alpha \cdot \text{sign}(w)$$
   - Leaves bias term unpenalized.

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