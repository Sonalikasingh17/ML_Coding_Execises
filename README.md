# 🚀 Machine Learning Coding Exercises

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)

A comprehensive collection of **50+ coding exercises** specifically designed for machine learning interviews and skill development. Each problem includes detailed solutions, test cases, and explanations to help you master both theoretical concepts and practical implementation skills. 
 
## 📖 Table of Contents 
   
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [How to Use This Repository](#how-to-use-this-repository)
- [Difficulty Levels](#difficulty-levels)
- [Topics Covered](#topics-covered)
- [Setup Instructions](#setup-instructions)
- [Running the Code](#running-the-code)
- [Contributing](#contributing)
- [Interview Preparation Tips](#interview-preparation-tips)
- [Additional Resources](#additional-resources)

## 📁 Repository Structure

```
ml-coding-exercises/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup
├── .gitignore                         # Git ignore rules
├── notebooks/                         # Jupyter notebooks
│   ├── 01_Python_Fundamentals.ipynb
│   ├── 02_NumPy_Pandas.ipynb
│   ├── 03_Scikit_Learn.ipynb
│   ├── 04_Neural_Networks.ipynb
│   ├── 05_TensorFlow_Keras.ipynb
│   └── 10_LeetCode_Style.ipynb
├── 01_python_fundamentals/
│   ├── README.md
│   ├── moving_average.py              # 🟢 Moving average calculator
│   ├── frequency_counter.py           # 🟢 Frequency tracking
│   └── tests/
│       └── test_fundamentals.py
├── 02_numpy_pandas/
│   ├── README.md
│   ├── matrix_operations.py           # 🟡 PCA, cosine similarity
│   ├── data_preprocessing.py          # 🟡 Data cleaning pipeline
│   └── tests/
│       └── test_numpy_pandas.py
├── 03_scikit_learn/
│   ├── README.md
│   ├── kmeans_from_scratch.py         # 🟡 K-means clustering
│   ├── logistic_regression_scratch.py # 🔴 Logistic regression + regularization
│   └── tests/
│       └── test_scikit_learn.py
├── 04_neural_networks/
│   ├── README.md
│   ├── neural_network_scratch.py      # 🔴 Complete neural network
│   ├── activation_functions.py        # 🟡 Activation functions
│   └── tests/
│       └── test_neural_networks.py
├── 05_tensorflow_keras/
│   ├── README.md
│   ├── custom_training_loop.py        # 🔴 Advanced TensorFlow
│   ├── custom_layers.py               # 🟡 Custom layer implementations
│   └── tests/
│       └── test_tensorflow.py
├── 10_leetcode_style/
│   ├── README.md
│   ├── ml_leetcode_problems.py        # 🔴 Algorithm challenges
│   ├── sliding_window_problems.py     # 🟡 Time series algorithms
│   └── tests/
│       └── test_leetcode.py
└── utils/
    ├── __init__.py
    ├── data_generators.py             # Test data generation
    └── visualization.py               # Plotting utilities
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ml-coding-exercises.git
cd ml-coding-exercises
```

### 2. Set Up Environment
```bash
# Create virtual environment
python -m venv ml_exercises_env

# Activate environment
# On Windows:
ml_exercises_env\Scripts\activate
# On macOS/Linux: 
source ml_exercises_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 3. Run Your First Exercise
```bash
# Test a simple problem
python 01_python_fundamentals/moving_average.py

# Run all tests
python -m pytest tests/ -v

# Start Jupyter notebook
jupyter lab notebooks/
```

## 🎯 How to Use This Repository

### 📚 For Learning
1. **Start with your skill level** - Begin with fundamentals if you're new to ML
2. **Read the problem statement** - Each file has detailed docstrings
3. **Try solving first** - Attempt the problem before looking at solutions
4. **Study the solution** - Understand the approach and complexity
5. **Run the tests** - Verify your understanding with provided test cases

### 💼 For Interview Preparation
1. **Time yourself** - Practice coding under time constraints
2. **Explain your approach** - Practice verbalizing your thought process
3. **Focus on edge cases** - Consider boundary conditions
4. **Optimize solutions** - Think about time/space complexity improvements
5. **Review regularly** - Revisit problems you found challenging

### 🔬 For Skill Development
1. **Implement variations** - Modify problems to test different scenarios
2. **Add your own tests** - Write additional test cases
3. **Compare with libraries** - See how your implementation compares to sklearn/TensorFlow
4. **Extend functionality** - Add features like visualization or logging

## 📊 Difficulty Levels

- 🟢 **Easy (20 problems)**: Basic concepts, straightforward implementations
  - Example: Moving average, basic data preprocessing
  
- 🟡 **Medium (20 problems)**: Moderate complexity, good ML understanding required
  - Example: PCA implementation, custom neural network layers
  
- 🔴 **Hard (15+ problems)**: Advanced topics, optimization challenges
  - Example: Complete neural network with backpropagation, custom training loops

## 📋 Topics Covered

### **Python Fundamentals** 🐍
- Data structures for ML
- Algorithm implementations
- Memory optimization
- Performance profiling

### **NumPy & Pandas** 📊
- Matrix operations and broadcasting
- Statistical computations
- Data manipulation and cleaning
- Performance optimization

### **Machine Learning Algorithms** 🤖
- Supervised learning (from scratch)
- Unsupervised learning (clustering)
- Model evaluation and validation
- Hyperparameter optimization

### **Deep Learning** 🧠
- Neural network architectures
- Backpropagation algorithm
- Optimization techniques
- Regularization methods

### **TensorFlow & Keras** 🔥
- Custom training loops
- Custom layers and models
- Advanced optimization
- Production deployment

### **Algorithm Challenges** 💡
- LeetCode-style problems for ML
- Dynamic programming for optimization
- Graph algorithms for neural networks
- String processing for NLP

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)
- 4GB+ RAM (for running neural network examples)

### Required Packages 
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
torch>=1.11.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
pytest>=7.0.0
```

### Optional Packages (for advanced exercises)
```txt
gym>=0.21.0                # Reinforcement learning
transformers>=4.15.0       # NLP transformers
opencv-python>=4.5.0       # Computer vision
plotly>=5.5.0              # Interactive visualization
```

## 🏃‍♂️ Running the Code

### Individual Problems
```bash
# Run a specific problem
python 01_python_fundamentals/moving_average.py

# Run with verbose output
python -v 02_numpy_pandas/matrix_operations.py
```

### Test Suites
```bash
# Run all tests
python -m pytest

# Run tests for specific module
python -m pytest 01_python_fundamentals/tests/ -v

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

### Jupyter Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Start classic Jupyter
jupyter notebook

# Convert notebook to script
jupyter nbconvert --to script notebooks/01_Python_Fundamentals.ipynb
```

### Performance Profiling
```bash
# Profile memory usage
python -m memory_profiler 04_neural_networks/neural_network_scratch.py

# Profile execution time
python -m cProfile -o profile.stats 03_scikit_learn/kmeans_from_scratch.py
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Adding New Problems
1. Fork the repository
2. Create a new branch: `git checkout -b feature/new-problem`
3. Add your problem following the existing format:
   - Include detailed docstring
   - Add comprehensive test cases
   - Update the README
4. Submit a pull request

### Problem Format Template
```python
"""
Problem: [Brief description]

[Detailed problem statement]

Requirements:
- [Requirement 1]
- [Requirement 2]

Example:
    Input: [example input]
    Output: [example output]

Time Complexity: O(?)
Space Complexity: O(?)
"""

class YourSolution:
    def solve(self, input_data):
        # Your implementation here
        pass

def test_your_solution():
    # Test cases here
    pass

if __name__ == "__main__":
    test_your_solution()
```

### Reporting Issues
- Use the GitHub issue tracker
- Include Python version and OS
- Provide minimal reproducible example
- Tag with appropriate labels

## 📝 Interview Preparation Tips

### Before the Interview
- [ ] Review basic ML concepts and formulas
- [ ] Practice coding without IDE assistance
- [ ] Time yourself on medium/hard problems
- [ ] Prepare questions about the role and company

### During the Interview
- [ ] Read the problem statement carefully
- [ ] Ask clarifying questions
- [ ] Start with a simple solution, then optimize
- [ ] Think out loud - explain your reasoning
- [ ] Test your solution with examples
- [ ] Discuss time/space complexity

### Common Interview Topics
1. **Algorithm Implementation**: K-means, linear regression, decision trees
2. **Data Preprocessing**: Handling missing values, feature scaling, encoding
3. **Model Evaluation**: Cross-validation, metrics, bias-variance tradeoffs
4. **System Design**: ML pipelines, scalability, monitoring
5. **Framework Usage**: TensorFlow/PyTorch custom implementations

## 📚 Additional Resources

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Pattern Recognition and Machine Learning" by Christopher Bishop
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville

### Online Courses
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)
- [CS229 Machine Learning - Stanford](http://cs229.stanford.edu/)
- [Deep Learning Specialization - Coursera](https://www.coursera.org/specializations/deep-learning)

### Practice Platforms
- [Kaggle Learn](https://www.kaggle.com/learn)
- [LeetCode](https://leetcode.com/problemset/all/?topicSlugs=machine-learning)
- [HackerRank AI](https://www.hackerrank.com/domains/ai)

### Communities
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [ML Twitter](https://twitter.com/search?q=%23MachineLearning)
- [Towards Data Science](https://towardsdatascience.com/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by common ML interview questions from top tech companies
- Built with contributions from the ML community
- Special thanks to open-source libraries: NumPy, Pandas, scikit-learn, TensorFlow, PyTorch

## 📞 Contact

- **Author**: Sonalika Singh
- **Email**: singhsonalika5@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/sonalika-singh-994a151a8/]
- **GitHub**: [https://github.com/Sonalikasingh17]

---

⭐ **Star this repository** if you find it helpful for your ML interview preparation!

🔄 **Fork and contribute** to help other ML practitioners succeed!

📢 **Share with friends** who are preparing for ML interviews!

---

*Happy Coding and Good Luck with Your Interviews! 🚀*
