if __name__ != "__main__":
  from algos.SVM import HardMarginSVM
else:
  from SVM import HardMarginSVM
import numpy as np


def krank(X, y, modelCall = HardMarginSVM):
  """
  One versus all except instead that k-1 binary classifiers are trained to rank ordinal categories
  """
  k = min(y), max(y) + 1
  models = []
  for i in range(*k):
    y_binary = np.array(1 if y_i > i else -1 for y_i in y)
    model = modelCall()
    model.fit(X, y_binary)
    models.append(model)

  def predict(X):
    scores = np.zeros((X.shape[0], k))
    for i, model in enumerate(models):
      scores[:, i + 1] = model.predict_proba(X)[:, 1]
    return np.argmax(scores, axis=1)

  return predict
