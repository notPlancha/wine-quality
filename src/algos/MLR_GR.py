from icecream import ic
from typing import Optional
from baseline import Model
import numpy as np

class MLR_GR(Model):
  theta: np.ndarray
  def fit(self, X, y, learning_rate=0.01, iterations=1000, tol=0.0001):
    X = super().fit(X, y)
    ic(X, y)
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
  
if __name__ == "__main__":
  # test mamae
  from imblearn.metrics import macro_averaged_mean_absolute_error as MAMAE
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  import pandas as pd
  X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = MLR_GR()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  ic(type(y_test), type(y_pred))
  # transform to series to avoid error
  y_test = pd.Series(y_test)
  y_pred = pd.Series(y_pred)
  print(f"Mean Absolute Error: {MAMAE(y_test, np.round(y_pred))}")