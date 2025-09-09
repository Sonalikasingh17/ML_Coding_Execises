"""
Problem: Implement custom training loop in TensorFlow with advanced features.

Features:
- Custom training step with tf.GradientTape
- Learning rate scheduling
- Gradient clipping
- Mixed precision training
- Custom metrics tracking
"""

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
        """
        Initialize custom trainer.
        
        Args:
            model: Keras model to train
            optimizer: Optimizer for training
            loss_fn: Loss function
            metrics: List of metrics to track
            gradient_clip_norm: Gradient clipping norm
            mixed_precision: Whether to use mixed precision training
        """
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
        """Execute one training step."""
        
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
        """Execute one validation step."""
        
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
        """
        Train the model with custom training loop.
        
        Returns:
            Dictionary containing training history
        """

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
    """Create a simple CNN model for testing."""
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
    """Test the custom trainer implementation."""

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
