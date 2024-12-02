import numpy as np
from cvxopt import matrix, solvers
from algos.baseline import Baseline


class SVR(Baseline):

  def __init__(self, C=1.0, epsilon=0.1, gamma=0.1):
    self.C = C
    self.epsilon = epsilon
    self.gamma = gamma
    super().__init__()

  def rbf_kernel(self, X1, X2):
    # RBF kernel between two matrices
    sq_dists = (
        np.sum(X1**2, axis=1).reshape(-1, 1)
        + np.sum(X2**2, axis=1)
        - 2 * np.dot(X1, X2.T)
    )
    return np.exp(-self.gamma * sq_dists)

  def fit(self, X, y):
    n_samples = X.shape[0]

    # Compute RBF kernel
    K = self.rbf_kernel(X, X)

    # Create matrices for quadratic optimization
    P = matrix(np.block([[K, -K], [-K, K]]))
    q = matrix(self.epsilon + np.hstack([y, -y]))

    G = matrix(np.block([[-np.eye(2 * n_samples)], [np.eye(2 * n_samples)]]))
    h = matrix(np.hstack([np.zeros(2 * n_samples), self.C * np.ones(2 * n_samples)]))

    A = matrix(np.ones((1, 2 * n_samples)))
    b = matrix(np.zeros(1))

    # Solve the optimization problem
    sol = solvers.qp(P, q, G, h, A, b)
    lambdas = np.array(sol["x"]).flatten()

    # Dual coefficients
    self.alpha = lambdas[:n_samples] - lambdas[n_samples:]
    self.X_train = X
    self.y_train = y

  def predict(self, X):
    # Prediction based on trained data and RBF kernel
    K_test = self.rbf_kernel(X, self.X_train)
    return np.dot(K_test, self.alpha) + np.mean(
        self.y_train - np.dot(self.rbf_kernel(self.X_train, self.X_train), self.alpha)
    )

if __name__ == "__main__":
  from sklearn.datasets import make_regression
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import mean_squared_error

  X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  svr = SVR(C=1.0, epsilon=0.1, gamma=0.1)
  svr.fit(X_train, y_train)
  y_pred = svr.predict(X_test)

  print("MSE:", mean_squared_error(y_test, y_pred))