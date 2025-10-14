import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    n_test = int(np.ceil(test_size * n))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standardize(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    if mean is None: mean = X.mean(axis=0)
    if std is None: std = X.std(axis=0) + 1e-8
    return (X - mean)/std, mean, std

def make_regression(n_samples=200, n_features=1, noise=1.0, random_state=None):
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    w_true = rng.randn(n_features)
    y = X @ w_true + rng.randn(n_samples)*noise
    return X, y, w_true

def make_classification(n_samples=300, random_state=None):
    rng = np.random.RandomState(random_state)
    # 2D blobs
    mean_shift = 2.5
    cov = np.array([[1.0, 0.3],[0.3, 1.0]])
    n0 = n_samples//2
    n1 = n_samples - n0
    X0 = rng.multivariate_normal([-mean_shift, -mean_shift], cov, size=n0)
    X1 = rng.multivariate_normal([ mean_shift,  mean_shift], cov, size=n1)
    X = np.vstack([X0, X1])
    y = np.array([0]*n0 + [1]*n1)
    return X, y