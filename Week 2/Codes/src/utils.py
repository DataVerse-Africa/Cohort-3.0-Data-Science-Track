"""
Utility Functions for Machine Learning

This module provides essential utility functions commonly used in machine learning workflows.
These functions support data preprocessing, dataset splitting, and synthetic data generation.

The module includes:
- Data Splitting: train_test_split for creating training and test sets
- Data Preprocessing: standardize for feature scaling
- Synthetic Data Generation: make_regression and make_classification for testing algorithms

These utilities are designed to be simple, efficient, and compatible with the machine learning
algorithms implemented in this package.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays or matrices into random train and test subsets.
    
    This function is essential for machine learning workflows as it allows you to
    evaluate model performance on unseen data. It randomly splits the dataset into
    training and testing portions while maintaining the correspondence between
    features and targets.
    
    Key Features:
    - Maintains correspondence between X and y
    - Optional shuffling for random sampling
    - Reproducible splits with random_state
    - Configurable test set size
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Target values or labels
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
        Should be between 0.0 and 1.0
    random_state : int, default=None
        Random seed for reproducible splits
        If None, the split will be different each time
    shuffle : bool, default=True
        Whether to shuffle the data before splitting
        If False, the test set will be the last test_size portion
        
    Returns:
    --------
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    y_train : array-like
        Training targets
    y_test : array-like
        Test targets
        
    Example:
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    """
    # Convert inputs to numpy arrays
    X = np.asarray(X)
    y = np.asarray(y)
    
    # Get number of samples
    n = X.shape[0]
    
    # Create index array
    idx = np.arange(n)
    
    # Shuffle indices if requested
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(idx)
    
    # Calculate number of test samples
    n_test = int(np.ceil(test_size * n))
    
    # Split indices
    test_idx = idx[:n_test]      # First n_test samples for testing
    train_idx = idx[n_test:]     # Remaining samples for training
    
    # Return split data
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X, mean=None, std=None):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Standardization (Z-score normalization) transforms features to have zero mean
    and unit variance. This is crucial for many machine learning algorithms that
    are sensitive to the scale of features.
    
    Mathematical Formula:
    X_std = (X - μ) / σ
    where μ is the mean and σ is the standard deviation
    
    Why Standardize:
    - Many algorithms assume features are on similar scales
    - Prevents features with larger scales from dominating
    - Improves convergence speed in gradient-based algorithms
    - Makes algorithms more stable and interpretable
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features to standardize
    mean : array-like, shape (n_features,), optional
        Mean values for each feature
        If None, computed from X
    std : array-like, shape (n_features,), optional
        Standard deviation values for each feature
        If None, computed from X
        
    Returns:
    --------
    X_std : array-like, shape (n_samples, n_features)
        Standardized features
    mean : array-like, shape (n_features,)
        Mean values used for standardization
    std : array-like, shape (n_features,)
        Standard deviation values used for standardization
        
    Example:
    --------
    >>> X_std, mean, std = standardize(X)
    >>> # For new data, use the same mean and std
    >>> X_new_std = (X_new - mean) / std
    """
    # Convert input to numpy array
    X = np.asarray(X, dtype=float)
    
    # Compute mean and std if not provided
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8  # Add small value to prevent division by zero
    
    # Standardize features
    X_std = (X - mean) / std
    
    return X_std, mean, std


def make_regression(n_samples=200, n_features=1, noise=1.0, random_state=None):
    """
    Generate a random regression problem.
    
    This function creates synthetic regression data with a linear relationship
    between features and target. It's useful for testing regression algorithms
    and understanding how they work on controlled data.
    
    Data Generation Process:
    1. Generate random features X from standard normal distribution
    2. Generate random true weights w_true
    3. Compute target: y = X @ w_true + noise
    
    Parameters:
    -----------
    n_samples : int, default=200
        Number of samples to generate
    n_features : int, default=1
        Number of features (dimensionality)
    noise : float, default=1.0
        Standard deviation of Gaussian noise added to the target
        Higher values = more noise = harder regression problem
    random_state : int, default=None
        Random seed for reproducible data generation
        
    Returns:
    --------
    X : array-like, shape (n_samples, n_features)
        Generated features
    y : array-like, shape (n_samples,)
        Generated target values
    w_true : array-like, shape (n_features,)
        True coefficients used to generate the data
        Useful for comparing with learned coefficients
        
    Example:
    --------
    >>> X, y, w_true = make_regression(n_samples=100, n_features=2, noise=0.5)
    >>> # Train a model and compare coefficients
    >>> model.fit(X, y)
    >>> print("True coefficients:", w_true)
    >>> print("Learned coefficients:", model.coef_)
    """
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Generate random features from standard normal distribution
    X = rng.randn(n_samples, n_features)
    
    # Generate random true weights
    w_true = rng.randn(n_features)
    
    # Generate target values with linear relationship and noise
    y = X @ w_true + rng.randn(n_samples) * noise
    
    return X, y, w_true


def make_classification(n_samples=300, random_state=None):
    """
    Generate a random binary classification problem.
    
    This function creates synthetic classification data with two classes that are
    generated from multivariate normal distributions. The classes are designed to
    be somewhat separable but with some overlap, making it a realistic classification
    problem.
    
    Data Generation Process:
    1. Generate two classes from multivariate normal distributions
    2. Class 0: centered at [-mean_shift, -mean_shift]
    3. Class 1: centered at [+mean_shift, +mean_shift]
    4. Both classes have the same covariance matrix with some correlation
    5. Combine the classes and create corresponding labels
    
    Parameters:
    -----------
    n_samples : int, default=300
        Total number of samples to generate
        Will be split roughly equally between classes
    random_state : int, default=None
        Random seed for reproducible data generation
        
    Returns:
    --------
    X : array-like, shape (n_samples, 2)
        Generated 2D features
    y : array-like, shape (n_samples,)
        Generated binary labels (0 or 1)
        
    Example:
    --------
    >>> X, y = make_classification(n_samples=200)
    >>> # Visualize the data
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(X[y==0, 0], X[y==0, 1], label='Class 0')
    >>> plt.scatter(X[y==1, 0], X[y==1, 1], label='Class 1')
    >>> plt.legend()
    >>> plt.show()
    """
    # Initialize random number generator
    rng = np.random.RandomState(random_state)
    
    # Define parameters for the two classes
    mean_shift = 2.5  # Distance between class centers
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])  # Covariance matrix with correlation
    
    # Calculate number of samples for each class
    n0 = n_samples // 2
    n1 = n_samples - n0
    
    # Generate samples for class 0 (negative class)
    X0 = rng.multivariate_normal([-mean_shift, -mean_shift], cov, size=n0)
    
    # Generate samples for class 1 (positive class)
    X1 = rng.multivariate_normal([mean_shift, mean_shift], cov, size=n1)
    
    # Combine features and labels
    X = np.vstack([X0, X1])
    y = np.array([0] * n0 + [1] * n1)
    
    return X, y