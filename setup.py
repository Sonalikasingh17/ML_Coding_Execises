from setuptools import setup, find_packages

setup(
    name="ML_Coding_Exercises",
    version="1.0.0",
    description="Comprehensive ML coding exercises for interview preparation",
    author="Sonalika Singh",
    author_email="singhsonalika5@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.8.0",
        "torch>=1.11.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "pytest>=7.0.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
