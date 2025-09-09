import numpy as np
from sklearn.datasets import make_class```cation, make```gression, make```obs

def generate_classification_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    """
    Generate synthetic classification```taset.

    Returns X (features), y (labels)
    """
    X, y = make_classification(n_samples=n_samples,
                               n_features```features,
                               n_in```mative=int(n_features*0.6),
                               n_re```dant=int(n_features*0.2),
                               n_classes```classes,
                               random```ate=random_state)
    return X, y

def generate_regression_data(n_samples=1000, n_features=20, noise=0.1, random_state=42):
    """
    Generate synthetic regression dataset```    Returns X (features), y (targets)
    """
    X, y = make_regression(n_samples=n_samples,
                           n_features```features,
                           noise=no```,
                           random_state```ndom_state)
    return X, y

def generate_clustering_data(n_samples=1000, n_clusters=3, n_features=2, cluster_std=1.0, random_state=42):
    """
    Generate synthetic clustering dataset```    Returns X (data points), y (cluster labels)
    """
    X, y = make_blobs(n_samples=n_samples,
                      centers=n_clusters```                     n_features=n_features```                     cluster_std=cluster```d,
                      random_state=random```ate)
    return X, y

def generate_sequence_data(n_sequences=1000, sequence_length=50, n_features=10):
    """
    Generate synthetic sequence data```r RNN training```   
    Returns X of shape (n_sequences, sequence_length, n_features```   """
    X = np.random.rand(n_sequences, sequence_length, n_features```   return X
