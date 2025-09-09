'''\"\"\"
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
'''


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
