"""Problem: Implement common matrix operations used in machine learning.

Tasks:
1. Matrix multiplication with broadcasting
2. Eigenvalue decomposition
3. SVD for dimensionality reduction
4. Cosine similarity calculation
"""

import numpy as np
from typing import Tuple

class MatrixOperations:
    
    @staticmethod
    def batch_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Multiply batches of matrices using broadcasting.
        A: (batch_size, m, k)
        B: (batch_size, k, n) or (k, n)
        Returns: (batch_size, m, n)
        """

        return np.matmul(A, B)
    
    @staticmethod
    def pca_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform PCA using SVD.
        X: (n_samples, n_features)
        Returns: (transformed_data, components)
        """

        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Perform SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Select top n_components
        components = Vt[:n_components]
        transformed = X_centered @ components.T
        
        return transformed, components
    
    @staticmethod
    def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        X: (n_samples, n_features)
        Returns: (n_samples, n_samples)
        """

        # Normalize vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity = X_norm @ X_norm.T
        
        return similarity
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Numerically stable softmax implementation.
        """

        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# Test cases
def test_matrix_operations():
    ops = MatrixOperations()
    
    # Test batch matrix multiplication
    A = np.random.randn(5, 3, 4)
    B = np.random.randn(5, 4, 2)
    result = ops.batch_matrix_multiply(A, B)
    assert result.shape == (5, 3, 2)
    
    # Test PCA
    X = np.random.randn(100, 10)
    transformed, components = ops.pca_transform(X, 3)
    assert transformed.shape == (100, 3)
    assert components.shape == (3, 10)
    
    # Test cosine similarity
    X = np.random.randn(5, 10)
    sim_matrix = ops.cosine_similarity_matrix(X)
    assert sim_matrix.shape == (5, 5)
    assert np.allclose(np.diag(sim_matrix), 1.0)  # Self-similarity should be 1
    
    # Test softmax
    x = np.array([[1, 2, 3], [4, 5, 6]])
    probs = ops.softmax(x, axis=1)
    assert np.allclose(np.sum(probs, axis=1), 1.0)
    
    print("All matrix operations tests passed!")

if __name__ == "__main__":
    test_matrix_operations()
