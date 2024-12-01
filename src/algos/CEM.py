import pandas as pd
import numpy as np
from pandas.core.series import Series

def CEM(y_real: Series, y_pred: Series) -> float:
  def prox(k_1, k_2):
    return -np.log2(
      (y_real[y_real.eq(k_1)].count()/2 + y_real[y_real.between(k_1, k_2, inclusive="right")].count()) / y_real.count()
    )
  def CM(k_1, k_2):
    return y_real[y_real.eq(k_1) & y_pred.eq(k_2)].count()
  
  return sum(
    prox(k_1, k_2) * CM(k_1, k_2)
    for k_1 in y_real.unique()
    for k_2 in y_real.unique()
  )/sum(
    prox(k, k) * y_real[y_real.eq(k)].count()
    for k in y_real.unique()
  )