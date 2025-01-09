import numpy as np
import pandas as pd
from baseline import Model
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from icecream import ic

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
  w: pd.Series
  b: float
  A: float
  B: float

  def fit(
      self,
      features: pd.DataFrame,
      target: pd.Series,
      w0: pd.Series | np.ndarray | None = None,
      gen_probs: bool = True,
  ):
    # target ∈ {-1, 1}
    if w0 is None:
      w0 = np.zeros(features.shape[1])  # or pd.Series(np.zeros(input.shape[1]))

    # objective: minimize_w,b: 1/2 * ||w||^2
    def objective(theta):
      w = theta[:-1]
      return 0.5 * np.dot(w, w)

    # s.t. y_i * (w x_i + b) >= 1
    def constraint(theta, i):
      w, b = theta[:-1], theta[-1]
      # pd.DataFrame call is important in case input is a numpy array
      return target[i] * (np.dot(w, pd.DataFrame(features).iloc[i] + b)) - 1 

    # add bias term
    # input = input.assign(bias=1)
    theta = np.append(w0, 0)

    constraints = [
        {"type": "ineq", "fun": constraint, "args": (i,)} for i in range(len(features))
    ]

    # minimize objective
    result = minimize(objective, theta, constraints=constraints, method="SLSQP")
    self.w = pd.Series(result.x[:-1])
    self.b = result.x[-1]

    if gen_probs:
      self.A, self.B = self.generate_A_B(self._get_decision_values(features), target, prior1=1, prior0=1)
    super().fit()

  def predict(self, input: pd.DataFrame) -> pd.Series:
    super().predict()
    decision_values = self._get_decision_values(input)
    return pd.Series(np.sign(decision_values), index=input.index)

  def predict_proba(self, input: pd.DataFrame) -> pd.Series:
    # https://link.springer.com/article/10.1007/s10994-007-5018-6
    if not hasattr(self, "A") or not hasattr(self, "B"):
      raise AttributeError("Model not trained with probabilities")
    super().predict()
    decision_values = self._get_decision_values(input)
    return pd.Series(1 / (1 + np.exp(self.A * decision_values + self.B)), index=input.index)
  
  def _get_decision_values(self, input: pd.DataFrame) -> pd.Series:
    # h(x) = sign(w^T x + b)
    return pd.Series(
        [np.dot(self.w, input.iloc[i]) + self.b for i in range(input.shape[0])],
        index=input.index,
    )
  @staticmethod
  def generate_A_B(deci, label, prior1, prior0) -> tuple[float, float]:
    # https://www.csie.ntu.edu.tw/~cjlin/papers/plattprob.pdf
    # Parameter setting
    maxiter = 100
    minstep=1e-10
    sigma=1e-12
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
        print("Line search fails")
        break
    if it >= maxiter:
      print("Reaching maximum iterations")
    return A, B
if __name__ == "__main__":

  from sklearn.datasets import make_classification
  from icecream import ic
  np.random.seed(0)
  
  X, y = make_classification()
  # change y to {-1, 1}
  y = pd.Series(y).apply(lambda x: 1 if x == 1 else -1)
  model = HardMarginSVM()
  model.fit(pd.DataFrame(X), pd.Series(y))
  predictions = model.predict(pd.DataFrame(X))
  probabilities = model.predict_proba(pd.DataFrame(X))
  
  ic(predictions)
  ic(probabilities)
  
  # Verify alignment
  alignment = (predictions == 1) == (probabilities >= 0.5)
  ic(alignment.all())