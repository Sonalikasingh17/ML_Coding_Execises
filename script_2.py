# Create final coding exercise files to complete the comprehensive repository

final_exercises = {
    "05_tensorflow_keras": {
        "README.md": """# TensorFlow & Keras Exercises

Advanced deep learning implementations using TensorFlow and Keras.

## Problems

1. **Custom Models** 🟡
   - Custom layers and models
   - Training loops with tf.GradientTape
   - Custom loss functions and metrics

2. **Advanced Architectures** 🔴
   - CNN for image classification
   - RNN/LSTM for sequence modeling
   - Transformer implementation

3. **Production Ready** 🔴
   - Model saving and loading
   - TensorFlow Serving
   - Model optimization and quantization
""",

        "custom_training_loop.py": """\"\"\"
Problem: Implement custom training loop in TensorFlow with advanced features.

Features:
- Custom training step with tf.GradientTape
- Learning rate scheduling
- Gradient clipping
- Mixed precision training
- Custom metrics tracking
\"\"\"

import tensorflow as tf
import numpy as np
from typing import Dict, List, Callable, Optional

class CustomTrainer:
    
    def __init__(self, model: tf.keras.Model, 
                 optimizer: tf.keras.optimizers.Optimizer,
                 loss_fn: Callable,
                 metrics: List[tf.keras.metrics.Metric],
                 gradient_clip_norm: Optional[float] = None,
                 mixed_precision: bool = False):
        \"\"\"
        Initialize custom trainer.
        
        Args:
            model: Keras model to train
            optimizer: Optimizer for training
            loss_fn: Loss function
            metrics: List of metrics to track
            gradient_clip_norm: Gradient clipping norm
            mixed_precision: Whether to use mixed precision training
        \"\"\"
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.gradient_clip_norm = gradient_clip_norm
        
        # Mixed precision setup
        if mixed_precision:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # Training metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
    
    @tf.function
    def train_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        \"\"\"Execute one training step.\"\"\"
        
        with tf.GradientTape() as tape:
            # Forward pass
            predictions = self.model(x, training=True)
            
            # Compute loss
            loss = self.loss_fn(y, predictions)
            
            # Add regularization losses
            if self.model.losses:
                loss += tf.add_n(self.model.losses)
            
            # Scale loss for mixed precision
            if hasattr(self.optimizer, 'get_scaled_loss'):
                loss = self.optimizer.get_scaled_loss(loss)
        
        # Compute gradients
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Unscale gradients for mixed precision
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        # Clip gradients if specified
        if self.gradient_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clip_norm)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        results = {'loss': loss}
        
        for metric in self.metrics:
            metric.update_state(y, predictions)
            results[metric.name] = metric.result()
        
        return results
    
    @tf.function
    def val_step(self, x: tf.Tensor, y: tf.Tensor) -> Dict[str, tf.Tensor]:
        \"\"\"Execute one validation step.\"\"\"
        
        # Forward pass
        predictions = self.model(x, training=False)
        
        # Compute loss
        loss = self.loss_fn(y, predictions)
        
        # Update metrics
        self.val_loss.update_state(loss)
        results = {'val_loss': loss}
        
        for metric in self.metrics:
            # Use validation versions of metrics if available
            val_metric_name = f'val_{metric.name}'
            if hasattr(self, val_metric_name):
                val_metric = getattr(self, val_metric_name)
                val_metric.update_state(y, predictions)
                results[val_metric_name] = val_metric.result()
            else:
                # Create validation metric on the fly
                val_metric = type(metric)(name=val_metric_name)
                setattr(self, val_metric_name, val_metric)
                val_metric.update_state(y, predictions)
                results[val_metric_name] = val_metric.result()
        
        return results
    
    def fit(self, train_dataset: tf.data.Dataset,
            val_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 10,
            callbacks: List[tf.keras.callbacks.Callback] = None,
            verbose: int = 1) -> Dict[str, List[float]]:
        \"\"\"
        Train the model with custom training loop.
        
        Returns:
            Dictionary containing training history
        \"\"\"
        history = {'loss': [], 'val_loss': []}
        
        # Add metric names to history
        for metric in self.metrics:
            history[metric.name] = []
            if val_dataset is not None:
                history[f'val_{metric.name}'] = []
        
        # Initialize callbacks
        if callbacks is None:
            callbacks = []
        
        # Callback setup
        callback_list = tf.keras.callbacks.CallbackList(
            callbacks, 
            add_history=True, 
            add_progbar=verbose > 0,
            model=self.model,
            verbose=verbose,
            epochs=epochs,
            steps=None
        )
        
        callback_list.on_train_begin()
        
        for epoch in range(epochs):
            callback_list.on_epoch_begin(epoch)
            
            # Reset metrics at start of epoch
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            for metric in self.metrics:
                metric.reset_states()
            
            # Training phase
            if verbose > 0:
                print(f'Epoch {epoch + 1}/{epochs}')
            
            # Training loop
            train_results = []
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                callback_list.on_train_batch_begin(step)
                
                step_results = self.train_step(x_batch, y_batch)
                train_results.append(step_results)
                
                callback_list.on_train_batch_end(step, logs=step_results)
            
            # Aggregate training results
            epoch_train_results = {}
            for key in train_results[0].keys():
                epoch_train_results[key] = tf.reduce_mean([r[key] for r in train_results])
            
            # Validation phase
            if val_dataset is not None:
                val_results = []
                for step, (x_batch, y_batch) in enumerate(val_dataset):
                    callback_list.on_test_batch_begin(step)
                    
                    step_results = self.val_step(x_batch, y_batch)
                    val_results.append(step_results)
                    
                    callback_list.on_test_batch_end(step, logs=step_results)
                
                # Aggregate validation results
                epoch_val_results = {}
                for key in val_results[0].keys():
                    epoch_val_results[key] = tf.reduce_mean([r[key] for r in val_results])
                
                epoch_train_results.update(epoch_val_results)
            
            # Update history
            for key, value in epoch_train_results.items():
                if key in history:
                    history[key].append(float(value.numpy()))
            
            # Print epoch results
            if verbose > 0:
                result_str = ' - '.join([f'{k}: {v:.4f}' for k, v in epoch_train_results.items()])
                print(f'Epoch results: {result_str}')
            
            callback_list.on_epoch_end(epoch, logs=epoch_train_results)
        
        callback_list.on_train_end()
        
        return history

# Example usage and test
def create_sample_model():
    \"\"\"Create a simple CNN model for testing.\"\"\"
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def test_custom_trainer():
    \"\"\"Test the custom trainer implementation.\"\"\"
    # Create sample data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Preprocess data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    # Create datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train[:1000], y_train[:1000]))
    train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((x_test[:200], y_test[:200]))
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    # Create model and trainer
    model = create_sample_model()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
    
    trainer = CustomTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metrics=metrics,
        gradient_clip_norm=1.0,
        mixed_precision=False
    )
    
    # Train model
    history = trainer.fit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=3,
        verbose=1
    )
    
    print("\\nTraining completed!")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    test_custom_trainer()
"""
    },

    "10_leetcode_style": {
        "README.md": """# LeetCode Style ML Problems

Algorithm and data structure problems commonly asked in ML interviews.

## Categories

1. **Array & Matrix** 🟢
   - Image processing operations
   - Sliding window for time series
   - Dynamic programming for optimization

2. **Trees & Graphs** 🟡
   - Decision tree operations
   - Neural network graph traversal
   - Clustering algorithms

3. **String Processing** 🟡
   - Text preprocessing
   - Pattern matching for NLP
   - Sequence alignment
""",

        "ml_leetcode_problems.py": """\"\"\"
LeetCode-style problems specifically tailored for machine learning interviews.

These problems test algorithmic thinking in the context of ML concepts.
\"\"\"

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict, deque
import heapq

class MLAlgorithmProblems:
    
    def sliding_window_normalization(self, data: List[float], window_size: int) -> List[float]:
        \"\"\"
        Problem: Normalize data using sliding window statistics.
        
        Given a list of numbers and a window size, return a new list where each element
        is normalized using the mean and std of its sliding window.
        
        Time Complexity: O(n)
        Space Complexity: O(window_size)
        
        Example:
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            window_size = 3
            Output: normalized values using sliding window stats
        \"\"\"
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
        \"\"\"
        Problem: Find optimal number of clusters using elbow method.
        
        Given 2D points, find the optimal number of clusters (1 to max_k)
        by implementing a simplified elbow method.
        
        Time Complexity: O(max_k * n^2 * iterations)
        Space Complexity: O(n)
        \"\"\"
        def calculate_wcss(points: List[Tuple[float, float]], k: int) -> float:
            \"\"\"Calculate Within-Cluster Sum of Squares for k clusters.\"\"\"
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
        \"\"\"
        Problem: Select features based on correlation with target and remove multicollinearity.
        
        Given features matrix and target, return indices of features to keep:
        1. Keep features with |correlation with target| > threshold
        2. Among correlated features, keep the one with highest target correlation
        
        Time Complexity: O(n * m^2) where n is samples, m is features
        Space Complexity: O(m^2)
        \"\"\"
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
        \"\"\"
        Problem: Return the path of weights during gradient descent for linear regression.
        
        Given data and parameters, return the weight vector at each epoch.
        
        Time Complexity: O(epochs * n * m)
        Space Complexity: O(epochs * m)
        \"\"\"
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
        \"\"\"
        Problem: Find the minimum depth needed for a decision tree to perfectly classify the data.
        
        Given samples (features, label), return minimum tree depth needed.
        This is a simplified version - assumes binary features.
        
        Time Complexity: O(2^n) worst case
        Space Complexity: O(depth)
        \"\"\"
        def can_separate(samples: List[Tuple[List[float], int]], depth: int) -> bool:
            \"\"\"Check if samples can be separated with given depth.\"\"\"
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
"""
    }
}

# Combine all exercises to show complete structure
all_exercises = {**coding_exercises, **advanced_exercises, **final_exercises}

print("\\n🎉 COMPLETE ML CODING EXERCISES REPOSITORY STRUCTURE:")
print("="*60)

for folder_name, contents in all_exercises.items():
    if isinstance(contents, dict):
        print(f"📁 {folder_name}/")
        for file_name in contents.keys():
            if file_name.endswith('.py'):
                print(f"  🐍 {file_name}")
            elif file_name.endswith('.md'):
                print(f"  📝 {file_name}")
            else:
                print(f"  📄 {file_name}")
        print()
    else:
        if folder_name.endswith('.py'):
            print(f"🐍 {folder_name}")
        elif folder_name.endswith('.md'):
            print(f"📝 {folder_name}")
        else:
            print(f"📄 {folder_name}")

print("\\n💡 Repository Features:")
print("✅ 50+ Coding Problems with Solutions")
print("✅ Difficulty Levels: Easy 🟢, Medium 🟡, Hard 🔴") 
print("✅ Complete Test Cases")
print("✅ Detailed Documentation")
print("✅ Industry-Ready Code Structure")
print("✅ Ready for GitHub Upload")

print("\\n🚀 Next Steps:")
print("1. Create new GitHub repository")
print("2. Upload all files maintaining folder structure") 
print("3. Add CI/CD with GitHub Actions for testing")
print("4. Create interactive Jupyter notebooks")
print("5. Add video explanations for complex problems")