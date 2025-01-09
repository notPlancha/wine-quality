import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from baseline import Model
from SVM import HardMarginSVM


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

  def fit(self, X: pd.DataFrame, y: pd.Series):
    self.k = (min(y), max(y) + 1)
    for i in range(*self.k):
      y_binary = np.array([1 if y_i > i else -1 for y_i in y])
      model = self.modelCall()
      model.fit(X, y_binary)
      self.classifiers.append(model)
    super().fit()

  def predict(self, X: pd.DataFrame):
    # https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf
    super().predict()
    return np.array([self.predict_one(x[1]) for x in X.iterrows()])
    
  def predict_one(self, x: pd.Series):
    probabilities = [1 - self.classifiers[0].predict(x)]
    return # CHECK probabilities
    for i in range(1, len(self.classifiers)):
      probabilities.append(
          self.classifiers[i - 1].predict(x) - self.classifiers[i].predict(x)
      )
  """
  def predict(self, X): # TODO fix
    # https://link.springer.com/content/pdf/10.1007/3-540-44795-4_13.pdf
    super().predict()
    predictions_per_model = []
    for model in self.classifiers:
      predictions_per_model.append(model.predict_proba(X))
    probabilities = [1 - predictions_per_model[0]]
    for i in range(1, len(predictions_per_model)):
      probabilities.append(predictions_per_model[i-1] * (1 - predictions_per_model[i]))
    probabilities.append(predictions_per_model[-1])
    # assert len(probabilities) == len(self.classifiers) + 1
    predictions_per_observation = np.argmax(probabilities, axis=0)
    return pd.Series(predictions_per_observation, index=X.index)
  """
if __name__ == "__main__":
  # Create a dummy dataset
  X, y = make_classification(
      n_samples=100,
      n_features=20,
      n_classes=3,
      n_informative=3,
      n_clusters_per_class=1,
      random_state=42,
  )

  # Initialize and fit the KRank model
  krank = KRank()
  krank.fit(X, y)

  # Make a prediction
  predictions = krank.predict(pd.DataFrame(X))
  print("Predictions:", predictions)
