# Create a comprehensive coding exercises structure for GitHub repository
import os
import json

# Define the repository structure and coding problems
coding_exercises = {
    "README.md": """# Machine Learning Coding Exercises 🚀

A comprehensive collection of coding exercises for machine learning interviews and skill development.

## Repository Structure

```
├── 01_python_fundamentals/
├── 02_numpy_pandas/
├── 03_scikit_learn/
├── 04_neural_networks/
├── 05_tensorflow_keras/
├── 06_pytorch/
├── 07_nlp_exercises/
├── 08_computer_vision/
├── 09_system_design/
└── 10_leetcode_style/
```

## How to Use

1. Start with fundamentals if you're a beginner
2. Each folder contains problems with increasing difficulty
3. Solutions are provided with detailed explanations
4. Test your solutions with provided test cases
5. Time yourself on coding challenges

## Difficulty Levels

- 🟢 **Easy:** Basic concepts and simple implementations
- 🟡 **Medium:** Moderate complexity requiring good understanding
- 🔴 **Hard:** Advanced topics and optimization challenges

## Quick Start

```bash
git clone https://github.com/Sonalikasingh17/ml-coding-exercises.git
cd ml-coding-exercises
pip install -r requirements.txt
```

Happy Coding! 💪
""",

    "requirements.txt": """numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
torch>=1.11.0
torchvision>=0.12.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
""",

    "01_python_fundamentals": {
        "README.md": """# Python Fundamentals for ML

Essential Python concepts needed for machine learning interviews.

## Problems

1. **Data Structures** 🟢
   - Implement a moving average calculator
   - Create a frequency counter
   - Build a simple LRU cache

2. **Algorithms** 🟡
   - K-means clustering from scratch
   - Linear regression implementation
   - Decision tree node splitting

3. **Advanced** 🔴
   - Custom gradient descent optimizer
   - Backpropagation implementation
   - Memory-efficient data loader
""",
        
        "moving_average.py": """\"\"\"
Problem: Implement a MovingAverage class that calculates moving average of numbers.

Example:
    ma = MovingAverage(3)
    ma.add(1) -> 1.0
    ma.add(2) -> 1.5  
    ma.add(3) -> 2.0
    ma.add(4) -> 3.0

Time Complexity: O(1) for add operation
Space Complexity: O(window_size)
\"\"\"

from collections import deque

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.queue = deque()
        self.sum = 0
    
    def add(self, num):
        if len(self.queue) == self.window_size:
            # Remove oldest element
            self.sum -= self.queue.popleft()
        
        # Add new element
        self.queue.append(num)
        self.sum += num
        
        return self.sum / len(self.queue)

# Test cases
def test_moving_average():
    ma = MovingAverage(3)
    assert ma.add(1) == 1.0
    assert ma.add(2) == 1.5
    assert ma.add(3) == 2.0
    assert ma.add(4) == 3.0
    assert ma.add(5) == 4.0
    print("All tests passed!")

if __name__ == "__main__":
    test_moving_average()
""",

        "frequency_counter.py": """\"\"\"
Problem: Create a frequency counter that tracks most common elements efficiently.

Requirements:
- Add elements in O(1) time
- Get most frequent element in O(1) time
- Handle ties by returning any of the most frequent elements

Example:
    fc = FrequencyCounter()
    fc.add('a')
    fc.add('b') 
    fc.add('a')
    fc.get_most_frequent() -> 'a'
\"\"\"

from collections import defaultdict

class FrequencyCounter:
    def __init__(self):
        self.counts = defaultdict(int)
        self.max_count = 0
        self.most_frequent = None
    
    def add(self, item):
        self.counts[item] += 1
        
        # Update most frequent if necessary
        if self.counts[item] > self.max_count:
            self.max_count = self.counts[item]
            self.most_frequent = item
    
    def get_most_frequent(self):
        return self.most_frequent
    
    def get_count(self, item):
        return self.counts[item]
    
    def get_all_counts(self):
        return dict(self.counts)

# Test cases
def test_frequency_counter():
    fc = FrequencyCounter()
    
    fc.add('a')
    assert fc.get_most_frequent() == 'a'
    
    fc.add('b')
    fc.add('a')
    assert fc.get_most_frequent() == 'a'
    assert fc.get_count('a') == 2
    
    fc.add('b')
    fc.add('b')
    assert fc.get_most_frequent() == 'b'
    assert fc.get_count('b') == 3
    
    print("All tests passed!")

if __name__ == "__main__":
    test_frequency_counter()
"""
    },

    "02_numpy_pandas": {
        "README.md": """# NumPy & Pandas Exercises

Data manipulation and numerical computing exercises essential for ML workflows.

## Problems

1. **NumPy Basics** 🟢
   - Matrix operations and broadcasting
   - Statistical calculations
   - Array reshaping and indexing

2. **Advanced NumPy** 🟡
   - Custom vectorized functions
   - Memory-efficient operations
   - Performance optimization

3. **Pandas Mastery** 🟡
   - Data cleaning and preprocessing
   - GroupBy operations
   - Time series analysis
""",

        "matrix_operations.py": """\"\"\"
Problem: Implement common matrix operations used in machine learning.

Tasks:
1. Matrix multiplication with broadcasting
2. Eigenvalue decomposition
3. SVD for dimensionality reduction
4. Cosine similarity calculation
\"\"\"

import numpy as np
from typing import Tuple

class MatrixOperations:
    
    @staticmethod
    def batch_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        \"\"\"
        Multiply batches of matrices using broadcasting.
        A: (batch_size, m, k)
        B: (batch_size, k, n) or (k, n)
        Returns: (batch_size, m, n)
        \"\"\"
        return np.matmul(A, B)
    
    @staticmethod
    def pca_transform(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"
        Perform PCA using SVD.
        X: (n_samples, n_features)
        Returns: (transformed_data, components)
        \"\"\"
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
        \"\"\"
        Compute pairwise cosine similarity matrix.
        X: (n_samples, n_features)
        Returns: (n_samples, n_samples)
        \"\"\"
        # Normalize vectors
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity = X_norm @ X_norm.T
        
        return similarity
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        \"\"\"
        Numerically stable softmax implementation.
        \"\"\"
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
""",

        "data_preprocessing.py": """\"\"\"
Problem: Implement common data preprocessing steps using pandas.

Tasks:
1. Handle missing values with different strategies
2. Encode categorical variables
3. Normalize/standardize numerical features
4. Feature engineering (date/time features)
\"\"\"

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Any

class DataPreprocessor:
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_stats = {}
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        \"\"\"
        Handle missing values with different strategies per column.
        
        strategy: dict mapping column names to strategies
        Strategies: 'drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill'
        \"\"\"
        df_processed = df.copy()
        
        for col, method in strategy.items():
            if col not in df_processed.columns:
                continue
                
            if method == 'drop':
                df_processed = df_processed.dropna(subset=[col])
            elif method == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif method == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            elif method == 'mode':
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            elif method == 'forward_fill':
                df_processed[col].fillna(method='ffill', inplace=True)
            elif method == 'backward_fill':
                df_processed[col].fillna(method='bfill', inplace=True)
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        \"\"\"
        Encode categorical features using label encoding.
        \"\"\"
        df_processed = df.copy()
        
        for col in categorical_cols:
            if col not in df_processed.columns:
                continue
                
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        return df_processed
    
    def normalize_features(self, df: pd.DataFrame, numerical_cols: List[str], 
                          method: str = 'standard') -> pd.DataFrame:
        \"\"\"
        Normalize numerical features.
        Methods: 'standard', 'minmax', 'robust'
        \"\"\"
        df_processed = df.copy()
        
        if method == 'standard':
            if self.scaler is None:
                self.scaler = StandardScaler()
                df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            else:
                df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def create_date_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        \"\"\"
        Extract useful features from datetime column.
        \"\"\"
        df_processed = df.copy()
        
        if date_col in df_processed.columns:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
            
            df_processed[f'{date_col}_year'] = df_processed[date_col].dt.year
            df_processed[f'{date_col}_month'] = df_processed[date_col].dt.month
            df_processed[f'{date_col}_day'] = df_processed[date_col].dt.day
            df_processed[f'{date_col}_dayofweek'] = df_processed[date_col].dt.dayofweek
            df_processed[f'{date_col}_quarter'] = df_processed[date_col].dt.quarter
            df_processed[f'{date_col}_is_weekend'] = (df_processed[date_col].dt.dayofweek >= 5).astype(int)
        
        return df_processed

# Test cases
def test_data_preprocessor():
    # Create sample data
    data = {
        'A': [1, 2, np.nan, 4, 5],
        'B': ['cat', 'dog', 'cat', np.nan, 'bird'],
        'C': [10, 20, 30, 40, 50],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
    }
    df = pd.DataFrame(data)
    
    preprocessor = DataPreprocessor()
    
    # Test missing value handling
    strategy = {'A': 'mean', 'B': 'mode'}
    df_clean = preprocessor.handle_missing_values(df, strategy)
    assert not df_clean['A'].isna().any()
    
    # Test categorical encoding
    df_encoded = preprocessor.encode_categorical_features(df_clean, ['B'])
    assert df_encoded['B'].dtype in [np.int32, np.int64]
    
    # Test date feature extraction
    df_dates = preprocessor.create_date_features(df_encoded, 'date')
    assert 'date_year' in df_dates.columns
    assert 'date_is_weekend' in df_dates.columns
    
    print("All data preprocessing tests passed!")

if __name__ == "__main__":
    test_data_preprocessor()
"""
    }
}

# Print the structure to show what we're creating
print("Created comprehensive ML coding exercises repository structure:")
print("📁 Repository Structure:")
for key in coding_exercises.keys():
    if isinstance(coding_exercises[key], dict):
        print(f"  📁 {key}/")
        for subkey in coding_exercises[key].keys():
            print(f"    📄 {subkey}")
    else:
        print(f"  📄 {key}")
