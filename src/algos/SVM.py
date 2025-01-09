import numpy as np
import pandas as pd
if __name__ != "__main__":
  from algos.baseline import Model
else:
  from baseline import Model
from scipy.optimize import minimize

class HardMarginSVM(Model):
  # https://link.springer.com/article/10.1007/s10462-018-9614-6
  """
  An infinite number of classifiers
  can be drawn for the
  given data but SVM finds the
  classifier with largest gap
  between support vectors.
  not allow mis-classification errors,
  that's why it is known as hard margin SVM
  """
  #

  def __init__(self):
    super().__init__()
    self.w: pd.Series  = None

  def fit(self, input: pd.DataFrame, target: pd.Series, w0: pd.Series | np.ndarray | None = None):
    # target ∈ {-1, 1}
    if w0 is None:
      w0 = np.zeros(input.shape[1]) # or pd.Series(np.zeros(input.shape[1]))

    # objective: minimize_w,b: 1/2 * ||w||^2
    def objective(theta):
      w = theta[:-1]
      return 0.5 * np.dot(w, w)
    
    # s.t. y_i * (w x_i + b) >= 1
    def constraint(theta, i):
      w, b = theta[:-1], theta[-1]
      return target[i] * (np.dot(w, input.iloc[i]) + b) - 1
    # add bias term
    # input = input.assign(bias=1)
    theta = np.append(w0, 0)

    constraints = [
      {"type": "ineq", "fun": constraint, "args": (i, )}
      for i in range(len(input))
    ]

    # minimize objective
    result = minimize(objective, theta, constraints=constraints, method="SLSQP")
    self.w = pd.Series(result.x[:-1])
    self.b = pd.Series(result.x[-1])
    super().fit()
  def predict(self, input: pd.DataFrame) -> pd.Series:
    super().predict()
    predictions = []
    for i in range(input.shape[0]):
      predictions.append(np.sign(np.dot(self.w, input.iloc[i]) + self.b))
    return pd.Series(predictions, index=input.index)

if __name__ == "__main__":

  from sklearn.datasets import make_classification
  from icecream import ic
  X, y = make_classification()
  # change y to {-1, 1}
  y = pd.Series(y).apply(lambda x: 1 if x == 1 else -1)
  model = HardMarginSVM()
  model.fit(pd.DataFrame(X), pd.Series(y))
  ic(model.predict(pd.DataFrame(X)))