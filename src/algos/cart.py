from baseline import Model
import numpy as np
from typing import Optional, Tuple
from icecream import ic

class Node:
    """A Node with a constructor"""
    def __init__(self, feature_index: Optional[int] = None, threshold: Optional[float] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None, value: Optional[float] = None):
        self.feature_index = feature_index  # Which feature index is used to split data
        self.threshold = threshold          # Threshold value for the split
        self.left = left                    # Left child node
        self.right = right                  # Right child node
        self.value = value                  # If node is leaf, store the predicted value(mean) in value

class CART(Model):
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, min_samples_leaf: int = 1):
        """ Constructor """
        self.max_depth = max_depth  # Maximum depth of tree, prevents overfitting
        self.min_samples_split = min_samples_split # Minimum number of samples to split a node
        self.min_samples_leaf = min_samples_leaf # Minimum number of samples required in a leaf
        self.root = None # Root of tree

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = super().fit(X, y)
        self.n_features = X.shape[1]  # Store number of features
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """ Recursively grow the tree by splitting the data."""
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return Node(value=np.mean(y))
        
        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # If no valid split is found, create a leaf node
        if best_feature is None:
            return Node(value=np.mean(y))
        
        # Create the split masks
        left_mask = X[:, best_feature] <= best_threshold
        
        # Split the data
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        # Check minimum samples in leaves
        if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
            return Node(value=np.mean(y))
        
        # Create node and recursively grow children
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        
        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )
        
    def _calculate_mse(self, y: np.ndarray) -> float:
        """ Calculate the Mean Squared Error (MSE) for a node."""
        if len(y) == 0:
            return 0.0
        return np.mean((y - np.mean(y)) ** 2)

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """ Find the best split that minimizes the weighted MSE."""
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        n_samples = len(y)
        
        # Iterate through all features
        for feature_idx in range(X.shape[1]):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            # Try all possible thresholds
            for threshold in thresholds:
                # Create masks for the split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Skip if split creates empty nodes
                if not left_mask.any() or not right_mask.any():
                    continue
                
                # Get the split data
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Calculate the weighted MSE
                n_left, n_right = len(y_left), len(y_right)
                mse = (n_left * self._calculate_mse(y_left) + 
                      n_right * self._calculate_mse(y_right)) / n_samples
                
                # Update best split if this is better
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = super().predict(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> float:
        """Traverse the tree to make a prediction for a single sample."""
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
      
if __name__ == '__main__':
    # Test the CART class
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    model = RegressionCART()
    model.fit(X, y)
    print(model.root)  # Should print the root node of the tree