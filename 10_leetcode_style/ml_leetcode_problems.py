"""
LeetCode-style problems specifically tailored for machine learning interviews.

These problems test algorithmic thinking in the context of ML concepts.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, deque
import heapq

class MLAlgorithmProblems:
    
    def sliding_window_normalization(self, data: List[float], window_size: int) -> List[float]:
        """
        Problem: Normalize data using sliding window statistics.
        
        Given a list of numbers and a window size, return a new list where each element
        is normalized using the mean and std of its sliding window.
        
        Time Complexity: O(n)
        Space Complexity: O(window_size)
        
        Example:
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            window_size = 3
            Output: normalized values using sliding window stats
        """

        if len(data) < window_size:
            return data
        
        result = []
        window = deque()
        window_sum = 0
        window_sum_sq = 0
        
        for i, num in enumerate(data):
            # Add current number to window
            window.append(num)
            window_sum += num
            window_sum_sq += num * num
            
            # Remove oldest element if window is full
            if len(window) > window_size:
                old_num = window.popleft()
                window_sum -= old_num
                window_sum_sq -= old_num * old_num
            
            # Calculate statistics and normalize
            if len(window) == window_size:
                mean = window_sum / window_size
                variance = (window_sum_sq / window_size) - (mean * mean)
                std = max(variance ** 0.5, 1e-8)  # Avoid division by zero
                
                normalized_val = (num - mean) / std
                result.append(normalized_val)
            else:
                result.append(num)  # Not enough data for normalization
        
        return result
    
    def find_optimal_clusters(self, points: List[Tuple[float, float]], 
                            max_k: int) -> int:
        """
        Problem: Find optimal number of clusters using elbow method.
        
        Given 2D points, find the optimal number of clusters (1 to max_k)
        by implementing a simplified elbow method.
        
        Time Complexity: O(max_k * n^2 * iterations)
        Space Complexity: O(n)
        """

        def calculate_wcss(points: List[Tuple[float, float]], k: int) -> float:
            """Calculate Within-Cluster Sum of Squares for k clusters."""

            if k >= len(points):
                return 0.0
            
            n = len(points)
            points_array = np.array(points)
            
            # Initialize centroids randomly
            indices = np.random.choice(n, k, replace=False)
            centroids = points_array[indices]
            
            # Run k-means for a few iterations
            for _ in range(10):
                # Assign points to nearest centroid
                distances = np.sqrt(((points_array[:, np.newaxis] - centroids) ** 2).sum(axis=2))
                assignments = np.argmin(distances, axis=1)
                
                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for i in range(k):
                    cluster_points = points_array[assignments == i]
                    if len(cluster_points) > 0:
                        new_centroids[i] = cluster_points.mean(axis=0)
                    else:
                        new_centroids[i] = centroids[i]
                
                centroids = new_centroids
            
            # Calculate WCSS
            wcss = 0
            for i in range(k):
                cluster_points = points_array[assignments == i]
                if len(cluster_points) > 0:
                    wcss += np.sum((cluster_points - centroids[i]) ** 2)
            
            return wcss
        
        # Calculate WCSS for different k values
        wcss_values = []
        for k in range(1, max_k + 1):
            wcss = calculate_wcss(points, k)
            wcss_values.append(wcss)
        
        # Find elbow using the "elbow method"
        if len(wcss_values) <= 2:
            return 1
        
        # Calculate rate of change
        diffs = []
        for i in range(1, len(wcss_values)):
            diffs.append(wcss_values[i-1] - wcss_values[i])
        
        # Find the point where improvement starts decreasing significantly
        max_improvement_drop = 0
        optimal_k = 1
        
        for i in range(1, len(diffs)):
            improvement_drop = diffs[i-1] - diffs[i]
            if improvement_drop > max_improvement_drop:
                max_improvement_drop = improvement_drop
                optimal_k = i + 1
        
        return optimal_k
    
    def feature_selection_correlation(self, features: List[List[float]], 
                                    target: List[float], 
                                    threshold: float = 0.7) -> List[int]:
        """
        Problem: Select features based on correlation with target and remove multicollinearity.
        
        Given features matrix and target, return indices of features to keep:
        1. Keep features with |correlation with target| > threshold
        2. Among correlated features, keep the one with highest target correlation
        
        Time Complexity: O(n * m^2) where n is samples, m is features
        Space Complexity: O(m^2)
        """

        features_array = np.array(features).T  # Shape: (n_samples, n_features)
        target_array = np.array(target)
        
        n_samples, n_features = features_array.shape
        
        # Calculate correlation with target
        target_correlations = []
        for i in range(n_features):
            corr = np.corrcoef(features_array[:, i], target_array)[0, 1]
            if np.isnan(corr):
                corr = 0
            target_correlations.append(abs(corr))
        
        # Filter features by target correlation
        candidate_features = []
        for i, corr in enumerate(target_correlations):
            if corr >= threshold:
                candidate_features.append(i)
        
        if len(candidate_features) <= 1:
            return candidate_features
        
        # Calculate feature-feature correlations
        selected_features = []
        remaining_features = candidate_features.copy()
        
        while remaining_features:
            # Find feature with highest target correlation
            best_feature = max(remaining_features, 
                             key=lambda x: target_correlations[x])
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            # Remove highly correlated features
            to_remove = []
            for feature in remaining_features:
                corr = np.corrcoef(features_array[:, best_feature], 
                                 features_array[:, feature])[0, 1]
                if np.isnan(corr):
                    corr = 0
                if abs(corr) > 0.8:  # High correlation threshold
                    to_remove.append(feature)
            
            for feature in to_remove:
                remaining_features.remove(feature)
        
        return selected_features
    
    def batch_gradient_descent_path(self, X: List[List[float]], y: List[float],
                                  learning_rate: float, epochs: int) -> List[List[float]]:
        """
        Problem: Return the path of weights during gradient descent for linear regression.
        
        Given data and parameters, return the weight vector at each epoch.
        
        Time Complexity: O(epochs * n * m)
        Space Complexity: O(epochs * m)
        """

        X_array = np.array(X)
        y_array = np.array(y)
        n_samples, n_features = X_array.shape
        
        # Add bias term
        X_with_bias = np.column_stack([np.ones(n_samples), X_array])
        
        # Initialize weights
        weights = np.zeros(n_features + 1)
        weight_path = [weights.copy()]
        
        for epoch in range(epochs):
            # Forward pass
            predictions = X_with_bias @ weights
            
            # Calculate gradients
            errors = predictions - y_array
            gradients = (1 / n_samples) * X_with_bias.T @ errors
            
            # Update weights
            weights -= learning_rate * gradients
            weight_path.append(weights.copy())
        
        return [w.tolist() for w in weight_path]
    
    def decision_tree_max_depth(self, samples: List[Tuple[List[float], int]]) -> int:
        """
        Problem: Find the minimum depth needed for a decision tree to perfectly classify the data.
        
        Given samples (features, label), return minimum tree depth needed.
        This is a simplified version - assumes binary features.
        
        Time Complexity: O(2^n) worst case
        Space Complexity: O(depth)
        """

        def can_separate(samples: List[Tuple[List[float], int]], depth: int) -> bool:
            """Check if samples can be separated with given depth."""

            if depth == 0:
                # Check if all samples have same label
                labels = [sample[1] for sample in samples]
                return len(set(labels)) <= 1
            
            if len(samples) <= 1:
                return True
            
            # Try splitting on each feature
            n_features = len(samples[0][0])
            
            for feature_idx in range(n_features):
                # Split samples based on feature value
                left_samples = []
                right_samples = []
                
                for sample in samples:
                    if sample[0][feature_idx] <= 0.5:  # Binary threshold
                        left_samples.append(sample)
                    else:
                        right_samples.append(sample)
                
                # Check if this split helps
                if len(left_samples) > 0 and len(right_samples) > 0:
                    if (can_separate(left_samples, depth - 1) and 
                        can_separate(right_samples, depth - 1)):
                        return True
            
            return False
        
        # Binary search for minimum depth
        max_depth = len(samples)
        
        for depth in range(1, max_depth + 1):
            if can_separate(samples, depth):
                return depth
        
        return max_depth

# Test cases
def test_ml_algorithm_problems():
    solver = MLAlgorithmProblems()
    
    # Test sliding window normalization
    print("Testing sliding window normalization...")
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    normalized = solver.sliding_window_normalization(data, 3)
    print(f"Original: {data[:5]}")
    print(f"Normalized: {normalized[:5]}")
    
    # Test optimal clusters
    print("\\nTesting optimal clusters...")
    points = [(i, i + np.random.random()) for i in range(20)]
    optimal_k = solver.find_optimal_clusters(points, 5)
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Test feature selection
    print("\\nTesting feature selection...")
    features = [[i, i*2, i*3 + np.random.random()] for i in range(100)]
    target = [i*2 + np.random.random() for i in range(100)]
    selected = solver.feature_selection_correlation(features, target, 0.5)
    print(f"Selected feature indices: {selected}")
    
    # Test gradient descent path
    print("\\nTesting gradient descent path...")
    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]
    path = solver.batch_gradient_descent_path(X, y, 0.01, 5)
    print(f"Initial weights: {path[0]}")
    print(f"Final weights: {path[-1]}")
    
    print("\\nAll tests completed!")

if __name__ == "__main__":
    test_ml_algorithm_problems()
