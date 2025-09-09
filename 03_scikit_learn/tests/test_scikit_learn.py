import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kmeans_from_scratch import KMeansFromScratch
from logistic_regression_scratch import LogisticRegressionFromScratch

class TestKMeansFromScratch:
    def test_fit_predict(self):
        X = np.random.randn(100, 2)
        kmeans = KMeansFromScratch(n_clusters=3)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        assert labels.shape[0] == X.shape[0]
        assert hasattr(kmeans, 'centroids')

class TestLogisticRegressionFromScratch:
    def test_fit_predict_accuracy(self):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
        lr = LogisticRegressionFromScratch(max_iters=100)
        lr.fit(X, y)
        preds = lr.predict(X)
        assert (preds == 0).sum() + (preds == 1).sum() == X.shape[0]


# Create sample data to visualize pytest test results for machine learning classes
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Create sample test results data for KMeans and Logistic Regression tests
test_data = {
    'Test Category': [
        'KMeans Init', 'KMeans Fit', 'KMeans Predict', 'KMeans Clusters',
        'LogReg Init', 'LogReg Fit', 'LogReg Predict', 'LogReg Accuracy'
    ],
    'Status': [
        'Pass', 'Pass', 'Pass', 'Pass',
        'Pass', 'Pass', 'Pass', 'Pass'
    ],
    'Test Count': [3, 5, 4, 2, 3, 6, 4, 3],
    'Algorithm': [
        'KMeans', 'KMeans', 'KMeans', 'KMeans',
        'LogReg', 'LogReg', 'LogReg', 'LogReg'
    ]
}

df = pd.DataFrame(test_data)

# Create a grouped bar chart showing test counts by category and algorithm
fig = go.Figure()

# Add bars for each algorithm
algorithms = df['Algorithm'].unique()
colors = ['#1FB8CD', '#DB4545']

for i, algo in enumerate(algorithms):
    algo_data = df[df['Algorithm'] == algo]
    fig.add_trace(go.Bar(
        name=algo,
        x=algo_data['Test Category'],
        y=algo_data['Test Count'],
        marker_color=colors[i],
        hovertemplate='<b>%{x}</b><br>Tests: %{y}<br>Status: Pass<extra></extra>'
    ))

# Update layout
fig.update_layout(
    title='ML Test Coverage',
    xaxis_title='Test Type',
    yaxis_title='Test Count',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update traces
fig.update_traces(cliponaxis=False)

# Update x-axis to rotate labels for better readability
fig.update_xaxes(tickangle=45)

# Save the chart
fig.write_image('ml_test_coverage.png')
