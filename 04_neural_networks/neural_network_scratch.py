"""
Problem: Implement a multi-layer neural network from scratch with backpropagation.

Features:
- Configurable architecture (hidden layers, neurons)
- Multiple activation functions (sigmoid, ReLU, tanh)
- Different optimizers (SGD, Momentum, Adam)
- Regularization techniques (L2, dropout)
"""

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
       """
        Initialize neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for optimization
            optimizer: Optimizer type ('sgd', 'momentum', 'adam')
            l2_reg: L2 regularization strength
            dropout_rate: Dropout probability (0 = no dropout)
        """
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
        """Initialize weights and biases using Xavier initialization."""
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
        """Initialize optimizer-specific parameters."""
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
        """Set activation function and its derivative."""
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
        """
        Forward propagation.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            training: Whether in training mode (for dropout)
            
        Returns:
            Output predictions
        """
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
        """
        Backward propagation.
        
        Args:
            X: Input data
            y: True labels
            predictions: Model predictions
        """
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
        """Update parameters using the chosen optimizer."""
        
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
        """Compute cost with regularization."""
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
        """Train the neural network."""
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
        """Make predictions on new data."""
        predictions = self.forward(X, training=False)
        return (predictions > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
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
