"""\"\"\"
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
"""

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
