import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from baseline import Model
from SVM import HardMarginSVM


class KRank(Model):
  """
  One versus all except that k-1 binary classifiers are trained to rank ordinal categories
  """

  def __init__(self, modelCall=HardMarginSVM):
    super().__init__()
    self.modelCall = modelCall
    self.models = []
    self.k = None

  def fit(self, X: pd.DataFrame, y: pd.Series):
    self.k = (min(y), max(y) + 1)
    for i in range(*self.k):
      y_binary = np.array([1 if y_i > i else -1 for y_i in y])
      model = self.modelCall()
      model.fit(X, y_binary)
      self.models.append(model)
    super().fit()

  def predict(self, X):
    super().predict()
    scores = np.zeros((X.shape[0], self.k[1]))
    for i, model in enumerate(self.models):
      scores[:, i + 1] = model.predict_proba(X)[:, 1]
    return np.argmax(scores, axis=1)



if __name__ == "__main__":
    # Create a dummy dataset
  X, y = make_classification(n_samples=100, n_features=20, n_classes=3, n_informative=3, n_clusters_per_class=1, random_state=42)
  
  # Initialize and fit the KRank model
  krank = KRank()
  krank.fit(X, y)
  
  # Make a prediction
  predictions = krank.predict(X)
  print("Predictions:", predictions)
