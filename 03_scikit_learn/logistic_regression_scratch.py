"""
Problem: Implement Logistic Regression with L1/L2 regularization from scratch.

Features:
- Binary and multiclass classification
- L1 (Lasso) and L2 (Ridge) regularization
- Gradient descent optimization
- Probability predictions
"""

import numpy as np
from typing import Optional

class LogisticRegressionFromScratch:
    
    def __init__(self, learning_rate: float = 0.01, max_iters: int = 1000, 
                 regularization: Optional[str] = None, reg_strength: float = 0.01,
                 fit_intercept: bool = True, tol: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.regularization = regularization  # None, 'l1', 'l2'
        self.reg_strength = reg_strength
        self.fit_intercept = fit_intercept
        self.tol = tol
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionFromScratch':
        """
        Fit logistic regression model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        """
        # Add intercept term if needed
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Store costs for convergence checking
        costs = []
        
        for i in range(self.max_iters):
            # Forward pass
            z = X @ self.weights
            predictions = self._sigmoid(z)
            
            # Calculate cost
            cost = self._calculate_cost(y, predictions)
            costs.append(cost)
            
            # Calculate gradients
            gradients = self._calculate_gradients(X, y, predictions)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Check for convergence
            if i > 0 and abs(costs[-2] - costs[-1]) < self.tol:
                print(f"Converged after {i + 1} iterations")
                break
        
        self.costs = costs
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        z = X @ self.weights
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _calculate_cost(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate logistic regression cost with regularization."""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Binary cross-entropy loss
        cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Add regularization
        if self.regularization == 'l1':
            # Don't regularize intercept (first weight)
            weights_to_reg = self.weights[1:] if self.fit_intercept else self.weights
            cost += self.reg_strength * np.sum(np.abs(weights_to_reg))
        elif self.regularization == 'l2':
            weights_to_reg = self.weights[1:] if self.fit_intercept else self.weights
            cost += self.reg_strength * np.sum(weights_to_reg ** 2)
        
        return cost
    
    def _calculate_gradients(self, X: np.ndarray, y: np.ndarray, 
                           predictions: np.ndarray) -> np.ndarray:
        """Calculate gradients for weights update."""
        n_samples = X.shape[0]
        
        # Basic gradient
        gradients = (1 / n_samples) * X.T @ (predictions - y)
        
        # Add regularization gradients
        if self.regularization == 'l1':
            l1_grad = self.reg_strength * np.sign(self.weights)
            if self.fit_intercept:
                l1_grad[0] = 0  # Don't regularize intercept
            gradients += l1_grad
        elif self.regularization == 'l2':
            l2_grad = 2 * self.reg_strength * self.weights
            if self.fit_intercept:
                l2_grad[0] = 0  # Don't regularize intercept
            gradients += l2_grad
        
        return gradients

# Test implementation
def test_logistic_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Generate test data
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different configurations
    configs = [
        {'regularization': None, 'name': 'No Regularization'},
        {'regularization': 'l1', 'reg_strength': 0.01, 'name': 'L1 Regularization'},
        {'regularization': 'l2', 'reg_strength': 0.01, 'name': 'L2 Regularization'}
    ]
    
    for config in configs:
        print(f"\\n=== {config['name']} ===\")
        
        # Custom implementation
        lr_custom = LogisticRegressionFromScratch(
            learning_rate=0.01, 
            max_iters=1000,
            regularization=config.get('regularization'),
            reg_strength=config.get('reg_strength', 0.01)
        )
        lr_custom.fit(X_train, y_train)
        
        # Predictions
        y_pred_custom = lr_custom.predict(X_test)
        y_proba_custom = lr_custom.predict_proba(X_test)
        
        print(f"Custom LR Accuracy: {accuracy_score(y_test, y_pred_custom):.4f}")
        print(f"Final cost: {lr_custom.costs[-1]:.4f}")

if __name__ == "__main__":
    test_logistic_regression()
