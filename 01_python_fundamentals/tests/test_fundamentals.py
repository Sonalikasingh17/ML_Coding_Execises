import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moving_average import MovingAverage
from frequency_counter import FrequencyCounter

class TestMovingAverage:
    def test_moving_average_basic(self):
        ma = MovingAverage(3)
        assert ma.add(1) == 1.0
        assert ma.add(2) == 1.5
        assert ma.add(3) == 2.0
        assert ma.add(4) == 3.0

class TestFrequencyCounter:
    def test_frequency_counter_basic(self):
        fc = FrequencyCounter()
        fc.add('a')
        fc.add('b')
        fc.add('a')
        assert fc.get_most_frequent() == 'a'
        assert fc.get_count('a') == 2

        

# Create a sample chart to demonstrate test coverage for MovingAverage and FrequencyCounter classes
import plotly.graph_objects as go
import plotly.express as px

# Sample data representing test coverage metrics for the two classes
classes = ['MovingAverage', 'FrequencyCounter']
test_types = ['Typical Cases', 'Edge Cases', 'Error Handling']

# Test coverage percentages for each class and test type
coverage_data = {
    'MovingAverage': [95, 88, 92],
    'FrequencyCounter': [98, 85, 90]
}

# Create a grouped bar chart showing test coverage
fig = go.Figure()

colors = ['#1FB8CD', '#DB4545', '#2E8B57']

for i, test_type in enumerate(test_types):
    fig.add_trace(go.Bar(
        name=test_type,
        x=classes,
        y=[coverage_data[class_name][i] for class_name in classes],
        marker_color=colors[i],
        text=[f"{coverage_data[class_name][i]}%" for class_name in classes],
        textposition='auto',
    ))

fig.update_traces(cliponaxis=False)

fig.update_layout(
    title='Test Coverage by Class',
    xaxis_title='Classes',
    yaxis_title='Coverage %',
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_yaxes(range=[0, 100])

fig.write_image('test_coverage_by_class.png')
