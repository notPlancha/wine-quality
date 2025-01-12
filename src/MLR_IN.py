# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:05:24 2025

@author: JoAnN
"""

# MLR HANDIN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

learning_rate=0.01
iterations=1000
tol=0.0001
def gradient_descent(X, y, eta, iterations, tol):
  m, n = X.shape  # Number of observations and features
  theta = np.zeros(n)  # Initialize weights
  for i in range(iterations):
    gradient = (2 / m) * (X.T @ ((X @ theta) - y))  #  gradient
    theta = theta - eta * gradient  # Uppdate weights

    if np.linalg.norm(gradient) < tol:
      print(f"Converged at iteration {i}")
      break
  return theta


def cross_validation(X, y, k_folds=5, learning_rate=0.01, iterations=1000, tol=0.0001):
  # Initialize metrics lists
  metrics_lists = {"mae": [], "accuracy": [], "mae_m": [], "cem_ord": []}

  fold_size = len(X) // k_folds

  for fold in range(k_folds):
    # Create validation indices
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size

    # Split data into training and validation
    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]
    X_train = np.concatenate([X[:val_start], X[val_end:]])
    y_train = np.concatenate([y[:val_start], y[val_end:]])

    # Scale the data (similar to your existing code)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Add bias term
    X_train_scaled = np.c_[np.ones(X_train_scaled.shape[0]), X_train_scaled]
    X_val_scaled = np.c_[np.ones(X_val_scaled.shape[0]), X_val_scaled]

    # Train model
    theta = gradient_descent(X_train_scaled, y_train, learning_rate, iterations, tol)

    # Make predictions
    y_pred = X_val_scaled @ theta

    # Calculate all metrics
    fold_metrics = evaluate_all_metrics(y_val, y_pred)

    # Store metrics
    for metric_name, value in fold_metrics.items():
      metrics_lists[metric_name].append(value)

  # Calculate means and standard deviations
  results = {}
  for metric_name, values in metrics_lists.items():
    results[f"{metric_name}_mean"] = np.mean(values)
    results[f"{metric_name}_std"] = np.std(values)

  return results