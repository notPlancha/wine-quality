from sklearn.metrics import mean_absolute_error
import numpy as np


def MAMAE(y_true, y_pred):
  # Get unique classes
  classes = np.unique(y_true)

  # Calculate MAE for each class
  class_maes = []
  for k in classes:
    # Get indices where true value is class k
    class_indices = y_true == k
    if sum(class_indices) > 0:  # Only if we have samples of this class
      mae_k = mean_absolute_error(y_true[class_indices], y_pred[class_indices])
      class_maes.append(mae_k)

  # Return average of class MAEs
  return np.mean(class_maes)


if __name__ == "__main__":
  from imblearn.metrics import macro_averaged_mean_absolute_error as calculate_mae_m

  y_true = np.array([0, 0, 1, 1, 2, 2])
  y_pred = np.array([0, 0, 2, 1, 2, 2])
  print(calculate_mae_m(y_true, y_pred))  # 0.3333333333333333
  print(MAMAE(y_true, y_pred))  # 0.3333333333333333
  print(MAMAE([0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1, 1]))
  print(calculate_mae_m([0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 0, 1, 1, 1]))
