from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):

  train_mean: float
  train_sd: float
  
  def __init__(self):
    self.trained: bool = False

  @abstractmethod
  def fit(self, X: np.array, y: np.array):

    self.train_mean = np.mean(X, axis=0)
    self.train_sd = np.std(X, axis=0)
    X = (X - self.train_mean) / self.train_sd    
    return X

  @abstractmethod
  def predict(self, X: np.array):
  
    X = (X - self.train_mean) / self.train_sd
    return X
  


class Baseline(Model):

  def __init__(self):
    self.mode = None
    super().__init__()

  def fit(self, input: pd.DataFrame, target: pd.Series):
    # saves the most frequent val
    X = super().fit(input.values, target.values)
    self.mode = target.mode().values[0]

  def predict(self, input: pd.DataFrame) -> pd.Series:
    X = super().predict(input.values)
    return pd.Series([self.mode] * len(input))