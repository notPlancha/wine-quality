from icecream import ic
from typing import Optional
from algos.baseline import Model
import numpy as np
from icecream import ic

class MLR_GR(Model):
  theta: np.ndarray
  def fit(self, X, y, learning_rate=0.01, iterations=1000, tol=0.0001):
    X = super().fit(X, y)
    m, n = X.shape  # Number of observations and features
    self.theta = np.zeros(n)  # Initialize weights
    for i in range(iterations):
      gradient = (2 / m) * (X.T @ ((X @ self.theta) - y))  #  gradient
      self.theta = self.theta - learning_rate * gradient  # Update weights

      if np.linalg.norm(gradient) < tol:
        ic(f"Converged at iteration {i}")
        break
  def predict(self, X):
    X = super().predict(X)
    ic(X, self.theta)
    return X @ self.theta
  