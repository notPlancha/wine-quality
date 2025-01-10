import numpy as np
from sklearn.datasets import fetch_openml
from baseline import Model
from SVM import HardMarginSVM
from icecream import ic

class KRank(Model):
  """
  https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf
  One versus all except that k-1 binary classifiers are trained to rank ordinal categories
  """

  def __init__(self, modelCall=HardMarginSVM):
    super().__init__()
    self.modelCall = modelCall
    self.classifiers: list = []
    self.k = None

  def fit(self, X: np.ndarray, y: np.ndarray):
    self.k = (min(y), max(y) + 1)
    for i in range(*self.k):
      y_binary = np.array([1 if y_i > i else -1 for y_i in y])
      model = self.modelCall()
      model.fit(X, y_binary)
      self.classifiers.append(model)
    super().fit()

  def predict(self, X: np.ndarray):
    # https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf
    # P(k0) = 1 - P(y > k0 | x)
    # P(ki) = P(y > ki-1 | x) * (1-P(y > ki | x)), 1 < i < k
    # P(kk) = P(y > kk-1 | x)
    super().predict()
    predictions = []
    for x in X:
      # transform x to shape (1, n_features)
      x = x.reshape(1, -1)
      probs: list[np.float32] = []
      probs.append(1 - self.classifiers[0].predict_proba(x)[0])
      for i in range(self.k[0] + 1, self.k[1]):
        probs.append(self.classifiers[i - self.k[0]].predict_proba(x)[0] * (1-probs[-1]))
      probs.append(self.classifiers[-1].predict_proba(x)[0])
      predictions.append(np.argmax(probs) + self.k[0])
    return np.array(predictions, dtype=np.int32)

if __name__ == "__main__":
  from pickle import dump, load
  import os
  # Load the wine-quality dataset from UCI ML repository
  data = fetch_openml(name='wine-quality-red', version=1)
  X, y = data.data, data.target
  X, y = np.array(X), np.array(y, dtype=np.int32)
  ic(X, y, type(X), type(y), X.shape, y.shape)
  # cache
  if not os.path.exists('krank.pkl'):
    krank = KRank()
    krank.fit(X, y)
    with open('krank.pkl', 'wb') as f:
      dump(krank, f)
  else:
    with open('krank.pkl', 'rb') as f:
      krank = load(f)
  # print krank json
  ic(krank.__dict__)
  ic(krank.classifiers[0].__dict__)
  predictions = krank.predict(X)
  ic(predictions, predictions.shape)
  
  ic(np.mean(predictions == y))
