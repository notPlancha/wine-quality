import numpy as np

def CEM(y_real: np.ndarray, y_pred: np.ndarray) -> float:
    def prox(k_1, k_2):
        return -np.log2(
            (
                np.sum(y_real == k_1) / 2
                + np.sum((y_real > k_1) & (y_real <= k_2))
            )
            / y_real.size
        )

    def CM(k_1, k_2):
        return np.sum((y_real == k_1) & (y_pred == k_2))

    unique_values = np.unique(y_real)
    return sum(
        prox(k_1, k_2) * CM(k_1, k_2)
        for k_1 in unique_values
        for k_2 in unique_values
    ) / sum(prox(k, k) * np.sum(y_real == k) for k in unique_values)
