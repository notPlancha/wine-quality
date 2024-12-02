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
    self.w:pd.Series  = None

  def fit(self,input: pd.DataFrame, target: pd.Series, w0: pd.Series | None = None):
    if w0 is None:
      w0 = np.zeros(input.shape[1]) # pd.Series(np.zeros(input.shape[1]))
    # 1/2 * ||w||^2
    objective = lambda w: (w @ w)/2
    # y_i * (w^T x_i) >= 1 for all i == -(y_i * (w^T x_i)) + 1 <= 0
    constraints = [
      {"type": "ineq", "fun": lambda w: -(target[i] * (input.iloc[i] @ w)) + 1}
      for i in range(len(input))
    ]
    # minimize 1/2 * ||w||^2
    result = minimize(objective, w0, constraints=constraints)
    self.w = pd.Series(result.x)
    # add bias term
    input = input.assign(bias=1) 
    

    super().fit()
    
if __name__ == "__main__":
  # test if     input = input.assign(bias=1)  works
  input = pd.DataFrame({"a": [1, 2, 3]})
  print(input)
  print(input.assign(bias=1))
  print(input) # should be the same as the original input
  print(pd.Series(np.zeros(input.shape[1])))

  # Define two vectors
  a = np.array([-1, 2, 3])
  b = np.array([4, 5, 6])

  # Compute the dot product using @
  dot_product = a @ b
  print(a @ b)
  print(np.linalg.norm(a)**2)
  print(a @ a)

  # test  constraint = lambda w: target * (input @ w)
  input = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
  target = pd.Series([1, -1, 1])
  w = np.array([1, 1])
  print(input @ w)
  print(target * (input @ w))

  def objective(x):
    return x[0] ** 2 + x[1] ** 2

  def constraint(x):
    return -(x[0] + x[1]) + 1  # This makes the inequality "<= 0"

  cons = {'type': 'ineq', 'fun': constraint}

  x0 = [0, 0]  # Initial guess
  res = minimize(objective, x0, constraints=cons)

  print(res)