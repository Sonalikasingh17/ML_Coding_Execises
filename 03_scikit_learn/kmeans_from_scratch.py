"""
Problem: Implement K-means clustering algorithm from scratch.

Requirements:
- Initialize centroids randomly
- Implement Lloyd's algorithm
- Handle convergence criteria
- Return cluster assignments and centroids

Time Complexity: O(n*k*i*d) where n=samples, k=clusters, i=iterations, d=dimensions
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class KMeansFromScratch:
    
    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        
    def fit(self, X: np.ndarray) -> 'KMeansFromScratch':
        """
        Fit K-means clustering to data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
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
        """Predict cluster labels for new data."""
        distances = self._calculate_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distances between points and centroids."""
        distances = np.zeros((X.shape[0], len(centroids)))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return distances
    
    def _calculate_inertia(self, X: np.ndarray) -> float:
        """Calculate within-cluster sum of squares."""
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
