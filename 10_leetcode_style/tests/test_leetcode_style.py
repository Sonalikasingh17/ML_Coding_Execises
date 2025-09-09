import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_leetcode_problems import MLAlgorithmProblems

class TestMLAlgorithmProblems:
    def test_sliding_window_normalization(self):
        solver = MLAlgorithmProblems()
        data = [1, 2, 3, 4, 5]
        normalized = solver.sliding_window_normalization(data, 3)
        assert len(normalized) == len(data)

    def test_find_optimal_clusters(self):
        solver = MLAlgorithmProblems()
        points = [(1,2), (2,3), (4,5), (10,10)]
        optimal_k = solver.find_optimal_clusters(points, 3)
        assert isinstance(optimal_k, int)
