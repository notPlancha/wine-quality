from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class Model(ABC):

  def __init__(self):
    self.trained: bool = False

  @abstractmethod
  def fit(self):
    self.trained = True

  @abstractmethod
  def predict(self):
    assert self.trained


class Baseline(Model):

  def __init__(self):
    self.mode = None
    super().__init__()

  def fit(self, input: pd.DataFrame, target: pd.Series):
    # saves the most frequent val
    self.mode = target.mode().values[0]
    super().fit()

  def predict(self, input: pd.DataFrame) -> pd.Series:
    super().predict()
    return pd.Series([self.mode] * len(input))