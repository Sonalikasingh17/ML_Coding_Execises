# Continue creating more advanced coding exercises

advanced_exercises = {
    "03_scikit_learn": {
        "README.md": """# Scikit-Learn Implementation Exercises

Implementing ML algorithms from scratch and using scikit-learn effectively.

## Problems

1. **Classical ML** 🟡
   - K-means clustering from scratch
   - Logistic regression with regularization
   - Decision tree implementation

2. **Ensemble Methods** 🔴
   - Random Forest from scratch
   - Gradient Boosting implementation
   - Voting classifier

3. **Model Evaluation** 🟢
   - Cross-validation strategies
   - Custom scoring functions
   - Hyperparameter optimization
""",

        "kmeans_from_scratch.py": """\"\"\"
Problem: Implement K-means clustering algorithm from scratch.

Requirements:
- Initialize centroids randomly
- Implement Lloyd's algorithm
- Handle convergence criteria
- Return cluster assignments and centroids

Time Complexity: O(n*k*i*d) where n=samples, k=clusters, i=iterations, d=dimensions
\"\"\"

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class KMeansFromScratch:
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        
    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        \"\"\"
        Fit K-means clustering to data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        \"\"\"
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroid
            distances = self._calculate_distances(X, self.centroids)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros((self.n_clusters, n_features))
            for k in range(self.n_clusters):
                if np.sum(self.labels == k) > 0:
                    new_centroids[k] = X[self.labels == k].mean(axis=0)
                else:
                    new_centroids[k] = self.centroids[k]
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, rtol=self.tol):
                print(f"Converged after {iteration + 1} iterations")
                break
                
            self.centroids = new_centroids
        
        # Calculate inertia (within-cluster sum of squares)
        self.inertia = self._calculate_inertia(X)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        \"\"\"Predict cluster labels for new data.\"\"\"
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        \"\"\"Calculate Euclidean distances between points and centroids.\"\"\"
        distances = np.zeros((X.shape[0], len(centroids)))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _calculate_inertia(self, X: np.ndarray) -> float:
        \"\"\"Calculate within-cluster sum of squares.\"\"\"
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[k]) ** 2)
        return inertia

# Test and comparison with sklearn
def test_kmeans_implementation():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    
    # Generate test data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                          random_state=42, cluster_std=0.60)
    
    # Our implementation
    kmeans_custom = KMeansFromScratch(n_clusters=4, max_iters=100)
    kmeans_custom.fit(X)
    
    # Sklearn implementation
    kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_sklearn.fit(X)
    
    # Compare inertia (should be similar)
    print(f"Custom K-means inertia: {kmeans_custom.inertia:.2f}")
    print(f"Sklearn K-means inertia: {kmeans_sklearn.inertia_:.2f}")
    
    # Test prediction on new data
    new_points = np.array([[0, 0], [1, 1]])
    custom_pred = kmeans_custom.predict(new_points)
    sklearn_pred = kmeans_sklearn.predict(new_points)
    
    print(f"Custom predictions: {custom_pred}")
    print(f"Sklearn predictions: {sklearn_pred}")
    
    print("K-means implementation test completed!")

if __name__ == "__main__":
    test_kmeans_implementation()
""",

        "logistic_regression_scratch.py": """\"\"\"
Problem: Implement Logistic Regression with L1/L2 regularization from scratch.

Features:
- Binary and multiclass classification
- L1 (Lasso) and L2 (Ridge) regularization
- Gradient descent optimization
- Probability predictions
\"\"\"

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
        \"\"\"
        Fit logistic regression model.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
        \"\"\"
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
        \"\"\"Predict class probabilities.\"\"\"
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        z = X @ self.weights
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        \"\"\"Predict binary class labels.\"\"\"
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        \"\"\"Sigmoid activation function with numerical stability.\"\"\"
        # Clip z to prevent overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def _calculate_cost(self, y: np.ndarray, predictions: np.ndarray) -> float:
        \"\"\"Calculate logistic regression cost with regularization.\"\"\"
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
        \"\"\"Calculate gradients for weights update.\"\"\"
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
"""
    },

    "04_neural_networks": {
        "README.md": """# Neural Networks from Scratch

Implementing neural networks and deep learning concepts from ground up.

## Problems

1. **Basic Neural Networks** 🟡
   - Multi-layer perceptron
   - Backpropagation algorithm
   - Different activation functions

2. **Advanced Architectures** 🔴
   - Convolutional Neural Networks
   - Recurrent Neural Networks
   - Attention mechanisms

3. **Optimization** 🔴
   - Different optimizers (SGD, Adam, RMSprop)
   - Batch normalization
   - Dropout regularization
""",

        "neural_network_scratch.py": """\"\"\"
Problem: Implement a multi-layer neural network from scratch with backpropagation.

Features:
- Configurable architecture (hidden layers, neurons)
- Multiple activation functions (sigmoid, ReLU, tanh)
- Different optimizers (SGD, Momentum, Adam)
- Regularization techniques (L2, dropout)
\"\"\"

import numpy as np
from typing import List, Callable, Optional, Tuple
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

class NeuralNetwork:
    
    def __init__(self, layer_sizes: List[int], activation: str = 'relu',
                 learning_rate: float = 0.01, optimizer: str = 'adam',
                 l2_reg: float = 0.0, dropout_rate: float = 0.0):
        \"\"\"
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type ('sgd', 'momentum', 'adam')
            l2_reg: L2 regularization strength
            dropout_rate: Dropout probability (0 = no dropout)
        \"\"\"
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # Initialize weights and biases
        self._initialize_parameters()
        
        # Initialize optimizer parameters
        self._initialize_optimizer()
        
        # Set activation function
        self._set_activation_function()
    
    def _initialize_parameters(self):
        \"\"\"Initialize weights and biases using Xavier initialization.\"\"\"
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            if self.activation == 'relu':
                # He initialization for ReLU
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            else:
                # Xavier initialization
                w = np.random.randn(fan_in, fan_out) * np.sqrt(1.0 / fan_in)
            
            b = np.zeros((1, fan_out))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _initialize_optimizer(self):
        \"\"\"Initialize optimizer-specific parameters.\"\"\"
        if self.optimizer == 'momentum':
            self.velocity_w = [np.zeros_like(w) for w in self.weights]
            self.velocity_b = [np.zeros_like(b) for b in self.biases]
            self.momentum = 0.9
        
        elif self.optimizer == 'adam':
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.beta1, self.beta2 = 0.9, 0.999
            self.epsilon = 1e-8
            self.t = 0  # time step
    
    def _set_activation_function(self):
        \"\"\"Set activation function and its derivative.\"\"\"
        if self.activation == 'sigmoid':
            self.activation_func = ActivationFunctions.sigmoid
            self.activation_derivative = ActivationFunctions.sigmoid_derivative
        elif self.activation == 'relu':
            self.activation_func = ActivationFunctions.relu
            self.activation_derivative = ActivationFunctions.relu_derivative
        elif self.activation == 'tanh':
            self.activation_func = ActivationFunctions.tanh
            self.activation_derivative = ActivationFunctions.tanh_derivative
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        \"\"\"
        Forward propagation.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output predictions
        \"\"\"
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Apply activation function
            if i < len(self.weights) - 1:  # Hidden layers
                a = self.activation_func(z)
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape)
                    a = a * dropout_mask / (1 - self.dropout_rate)
            else:  # Output layer (no activation for regression, sigmoid for binary classification)
                a = ActivationFunctions.sigmoid(z)  # For binary classification
            
            self.activations.append(a)
            current_input = a
        
        return self.activations[-1]
    
    def backward(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray):
        \"\"\"
        Backward propagation.
        
        Args:
            X: Input data
            y: True labels
            predictions: Model predictions
        \"\"\"
        m = X.shape[0]  # batch size
        
        # Initialize gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        dz = predictions - y
        
        # Backward pass through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradients for weights and biases
            dw[i] = (1/m) * np.dot(self.activations[i].T, dz)
            db[i] = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # Add L2 regularization to weights
            if self.l2_reg > 0:
                dw[i] += self.l2_reg * self.weights[i]
            
            # Propagate error to previous layer
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * \
                     self.activation_derivative(self.z_values[i-1])
        
        # Update parameters using chosen optimizer
        self._update_parameters(dw, db)
    
    def _update_parameters(self, dw: List[np.ndarray], db: List[np.ndarray]):
        \"\"\"Update parameters using the chosen optimizer.\"\"\"
        
        if self.optimizer == 'sgd':
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dw[i]
                self.biases[i] -= self.learning_rate * db[i]
        
        elif self.optimizer == 'momentum':
            for i in range(len(self.weights)):
                self.velocity_w[i] = self.momentum * self.velocity_w[i] + dw[i]
                self.velocity_b[i] = self.momentum * self.velocity_b[i] + db[i]
                
                self.weights[i] -= self.learning_rate * self.velocity_w[i]
                self.biases[i] -= self.learning_rate * self.velocity_b[i]
        
        elif self.optimizer == 'adam':
            self.t += 1
            
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dw[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
                
                # Update biased second raw moment estimate
                self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dw[i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
                
                # Compute bias-corrected first moment estimate
                m_w_corrected = self.m_w[i] / (1 - self.beta1 ** self.t)
                m_b_corrected = self.m_b[i] / (1 - self.beta1 ** self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_w_corrected = self.v_w[i] / (1 - self.beta2 ** self.t)
                v_b_corrected = self.v_b[i] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                self.weights[i] -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
    
    def compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Compute cost with regularization.\"\"\"
        m = y_true.shape[0]
        
        # Binary cross-entropy loss
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -(1/m) * np.sum(y_true * np.log(y_pred_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_clipped))
        
        # Add L2 regularization
        if self.l2_reg > 0:
            l2_cost = 0
            for w in self.weights:
                l2_cost += np.sum(w ** 2)
            cost += (self.l2_reg / (2 * m)) * l2_cost
        
        return cost
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            verbose: bool = True) -> List[float]:
        \"\"\"Train the neural network.\"\"\"
        costs = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X, training=True)
            
            # Compute cost
            cost = self.compute_cost(y, predictions)
            costs.append(cost)
            
            # Backward pass
            self.backward(X, y, predictions)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                accuracy = np.mean((predictions > 0.5) == y)
                print(f"Epoch {epoch}: Cost = {cost:.4f}, Accuracy = {accuracy:.4f}")
        
        return costs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        \"\"\"Make predictions on new data.\"\"\"
        predictions = self.forward(X, training=False)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        \"\"\"Get prediction probabilities.\"\"\"
        return self.forward(X, training=False)

# Test the implementation
def test_neural_network():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    
    # Generate test data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                             n_redundant=5, n_classes=2, random_state=42)
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape y for binary classification
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Test different configurations
    configs = [
        {'optimizer': 'sgd', 'activation': 'relu'},
        {'optimizer': 'momentum', 'activation': 'relu'},
        {'optimizer': 'adam', 'activation': 'relu'},
        {'optimizer': 'adam', 'activation': 'sigmoid', 'l2_reg': 0.01}
    ]
    
    for config in configs:
        print(f"\\n=== Testing {config} ===\")
        
        nn = NeuralNetwork(
            layer_sizes=[20, 64, 32, 1],
            learning_rate=0.001,
            **config
        )
        
        # Train the model
        costs = nn.fit(X_train_scaled, y_train, epochs=500, verbose=False)
        
        # Make predictions
        y_pred = nn.predict(X_test_scaled)
        y_proba = nn.predict_proba(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Final cost: {costs[-1]:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    test_neural_network()
"""
    }
}

print("\\nCreated additional advanced exercises:")
for folder, contents in advanced_exercises.items():
    print(f"📁 {folder}/")
    for file_name in contents.keys():
        print(f"  📄 {file_name}")