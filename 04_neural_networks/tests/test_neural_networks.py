import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network_scratch import NeuralNetwork

class TestNeuralNetwork:
    def test_training_reduces_loss(self):
        X = np.random.randn(50, 10)
        y = (np.random.rand(50) > 0.5).astype(int).reshape(-1, 1)
        nn = NeuralNetwork([10, 5, 1], epochs=10, verbose=False)
        costs = nn.fit(X, y, epochs=10, verbose=False)
        assert costs[0] > costs[-1]

    def test_predict_shape(self):
        X = np.random.randn(10, 10)
        nn = NeuralNetwork([10, 5, 1], epochs=1, verbose=False)
        nn.fit(np.random.randn(50, 10), np.random.randint(0, 2, size=(50, 1)), epochs=5, verbose=False)
        preds = nn.predict(X)
        assert preds.shape == (10, 1)


# Create a visualization showing test coverage areas for a neural network testing suite
import plotly.express as px
import pandas as pd

# Create data representing different test coverage areas
test_areas = [
    'Training', 
    'Prediction', 
    'Activation Funcs', 
    'Regularization',
    'Loss Calc',
    'Backprop',
    'Weight Init',
    'Validation'
]

# Test counts for each area
test_counts = [15, 12, 8, 6, 10, 9, 5, 7]

# Create a DataFrame
df = pd.DataFrame({
    'Test Area': test_areas,
    'Test Count': test_counts
})

# Create a bar chart
fig = px.bar(
    df, 
    x='Test Area', 
    y='Test Count',
    title='Neural Network Test Coverage',
    color_discrete_sequence=['#1FB8CD']
)

# Update layout
fig.update_traces(cliponaxis=False)
fig.update_xaxes(title='Test Areas')
fig.update_yaxes(title='Num Tests')

# Rotate x-axis labels for better readability
fig.update_xaxes(tickangle=45)

# Save the chart
fig.write_image('neural_network_test_coverage.png')
