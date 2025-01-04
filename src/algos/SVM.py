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

  def fit(self,input: pd.DataFrame, target: pd.Series, w0: pd.Series | np.ndarray | None = None):
    # target ∈ {-1, 1}
    if w0 is None:
      w0 = np.zeros(input.shape[1]) # pd.Series(np.zeros(input.shape[1]))

    # objective: 1/2 * ||w||^2
    objective = lambda w: (w @ w)/2
    # add bias term
    input = input.assign(bias=1)
    w0 = np.append(w0, 0)

    # ∀i: y_i * (w^T x_i) >= 1  == -(y_i * (w^T x_i)) + 1 <= 0 
    constraints = [
      {"type": "ineq", "fun": lambda w: -(target[i] * (input.iloc[i] @ w)) + 1}
      for i in range(len(input))
    ]
    # minimize objective
    # TODO: THE MAIN PROBLEM IS THAT THE OPTIMIZER IS FINISHING AFTER 1 ITERATION FSR
    result = ic(minimize(objective, w0, constraints=constraints))
    ic(result.success)
    ic(result.fun)
    self.w = pd.Series(result.x)
    
    super().fit()
  def predict(self, input: pd.DataFrame) -> pd.Series:
    super().predict()
    # h(x) = sign(w^T x)
    def sign(x):
      return 1 if x >= 0 else -1
    

if __name__ == "__main__":
  from sklearn.datasets import make_classification
  from icecream import ic
  X, y = make_classification()
  # change y to {-1, 1}
  y = pd.Series(y).apply(lambda x: 1 if x == 1 else -1)
  model = HardMarginSVM()
  model.fit(pd.DataFrame(X), pd.Series(y))
  ic(model.w)
  ic(model.predict(pd.DataFrame(X)))
