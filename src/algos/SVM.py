from warnings import warn
import numpy as np
import pandas as pd
from baseline import Model
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from icecream import ic
ic.configureOutput(includeContext=True)

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
  w: np.ndarray
  b: np.float64
  A: np.float64
  B: np.float64

  def fit(
      self,
      features: np.ndarray,
      target: np.ndarray,
      w0: np.ndarray | None = None,
  ):
    ic(features.shape, target.shape)
    # target ∈ {-1, 1}
    if w0 is None:
      w0 = np.zeros(features.shape[1])  # or pd.Series(np.zeros(input.shape[1]))

    # objective: minimize_w,b: 1/2 * ||w||^2
    def objective(theta: np.ndarray) -> float:
      w = theta[:-1]
      return 0.5 * np.dot(w, w)

    # s.t. y_i * (w x_i + b) >= 1
    def constraint(theta: np.ndarray, i: int) -> float:
      w, b = theta[:-1], theta[-1]
      return target[i] * (np.dot(w, features[i]) + b) - 1 

    # add bias term
    theta = np.append(w0, 0)

    constraints = [
        {"type": "ineq", "fun": constraint, "args": (i,)} for i in range(len(features))
    ]

    # minimize objective
    result = minimize(objective, theta, constraints=constraints, method="SLSQP")
    self.w = result.x[:-1]
    self.b = result.x[-1]

    self.A, self.B = self.generate_A_B(self._get_decision_values(features), target)
    super().fit()

  def predict(self, input: np.ndarray) -> np.ndarray:
    super().predict()
    probabilities = self.predict_proba(input)
    return np.where(probabilities >= 0.5, 1, -1)

  def predict_proba(self, input: np.ndarray) -> np.ndarray:
    """
    Return the probability of 1
    """
    # https://link.springer.com/article/10.1007/s10994-007-5018-6
    if not hasattr(self, "A") or not hasattr(self, "B"):
      raise AttributeError("Model not trained with probabilities")
    super().predict()
    ic(input.shape)
    decision_values = self._get_decision_values(input)
    ic(decision_values.shape)
    return 1 / (1 + np.exp(self.A * decision_values + self.B))
  
  def _get_decision_values(self, input: np.ndarray) -> np.ndarray:
    # h(x) = sign(w^T x + b)
    return np.array([np.dot(self.w, input[i]) + self.b for i in range(input.shape[0])])

  @staticmethod
  def generate_A_B(deci: np.ndarray, label: np.ndarray, maxiter=1000, minstep=1e-10, sigma=1e12) -> tuple[float, float]:
    """
    https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
    Generate A and B for Platt scaling
    :param deci: array of SVM decision values
    :param label: array of booleans: is the example labeled +1?
    """
    prior0 = sum(label == -1)
    prior1 = sum(label == 1)
    # Construct initial values: target support in array t
    #                           initial function value in fval
    hiTarget, loTarget = (prior1+1.0)/(prior1+2.0), 1/(prior0+2.0)
    len_=prior1+prior0 # Total number of data  
    t=np.zeros(len_)
    for i in range(len_):
      if label[i] > 0:
        t[i]=hiTarget
      else:
        t[i]=loTarget
    A, B, fval = 0.0, np.log((prior0+1.0)/(prior1+1.0)), 0.0
    for i in range(len_):
      fApB = deci[i]*A+B
      if fApB >= 0:
        fval += t[i]*fApB + np.log(1+np.exp(-fApB))
      else:
        fval += (t[i]-1)*fApB +np.log(1+np.exp(fApB))
    for it in range(maxiter):
      # Update Gradient and Hessian (use H' = H + sigma I)
      h11, h22, h21, g1, g2 = sigma, sigma, 0.0, 0.0, 0.0
      for i in range(len_):
        fApB = deci[i]*A+B
        if fApB >= 0:
          p = np.exp(-fApB)/(1.0+np.exp(-fApB))
          q = 1.0/(1.0+np.exp(-fApB))
        else:
          p = 1.0/(1.0+np.exp(fApB))
          q = np.exp(fApB)/(1.0+np.exp(fApB))
        d2 = p*q
        h11 += deci[i]*deci[i]*d2
        h22 += d2
        h21 += deci[i]*d2
        d1 = t[i]-p
        g1 += deci[i]*d1
        g2 += d1
      # Stopping criteria
      if abs(g1) < 1e-5 and abs(g2) < 1e-5:
        break
      # Compute modified Newton directions
      det = h11*h22 - h21*h21
      dA = -(h22*g1 - h21*g2)/det
      dB = -(-h21*g1+ h11*g2)/det
      gd = g1*dA + g2*dB
      stepsize = 1
      while stepsize >= minstep:
        newA = A + stepsize*dA
        newB = B + stepsize*dB
        newf = 0.0
        for i in range(len_):
          fApB = deci[i]*newA+newB
          if fApB >= 0:
            newf += t[i]*fApB + np.log(1+np.exp(-fApB))
          else:
            newf += (t[i]-1)*fApB + np.log(1+np.exp(fApB))
        if newf < fval + 0.0001*stepsize*gd:
          A, B, fval = newA, newB, newf
          break # Sufficient decrease satisfied
        else:
          stepsize /= 2
      if stepsize < minstep:
        warn("Line search fails")
        break
    if it >= maxiter:
      warn("Reaching maximum iterations")
    return A, B

if __name__ == "__main__":
  from sklearn.datasets import load_breast_cancer
  from icecream import ic
  np.random.seed(0)
  
  data = load_breast_cancer()
  X, y = data.data, data.target
  ic(X.shape, y.shape)
  # change y to {-1, 1}
  y = np.where(y == 1, 1, -1)
  model = HardMarginSVM()
  model.fit(X, y)
  probabilities = model.predict_proba(X)
  predictions = model.predict(X)
  ic(model.__dict__)
  ic(predictions)
  ic(probabilities)
  ic(probabilities[0])
  ic(sum(predictions == y) / len(y))