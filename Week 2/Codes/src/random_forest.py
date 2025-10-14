"""
Random Forest Implementation from Scratch

This module implements both Classification and Regression Random Forests from scratch.

Random Forest is an ensemble learning method that combines multiple decision trees to create
a more robust and accurate model. It uses two key techniques to reduce overfitting and
improve generalization:

1. Bootstrap Aggregating (Bagging): Each tree is trained on a random subset of the data
2. Feature Randomness: Each tree considers only a random subset of features for splitting

Key Concepts:
- Ensemble Method: Combines predictions from multiple models
- Bootstrap Sampling: Random sampling with replacement from training data
- Feature Subsampling: Random selection of features for each split
- Voting (Classification): Majority vote from all trees
- Averaging (Regression): Mean prediction from all trees

Advantages:
- Reduces overfitting compared to single decision trees
- Handles missing values well
- Provides feature importance estimates
- Works well with both numerical and categorical data
- Less sensitive to outliers than single trees

Disadvantages:
- Less interpretable than single decision trees
- Can be memory intensive for large datasets
- May not perform well on very sparse data
"""

import numpy as np
from .decision_tree import DecisionTreeClassifierScratch, DecisionTreeRegressorScratch


def _bootstrap_sample(X, y, rng):
    """
    Create a bootstrap sample from the training data.
    
    Bootstrap sampling is a resampling technique where we randomly sample with replacement
    from the original dataset to create a new dataset of the same size. This introduces
    randomness and diversity among the trees in the forest.
    
    Key Properties:
    - Sample size equals original dataset size
    - Sampling is done with replacement (some samples may appear multiple times)
    - Some samples may not appear at all (out-of-bag samples)
    - On average, ~63.2% of original samples appear in each bootstrap sample
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target values/labels
    rng : numpy.random.RandomState
        Random number generator for reproducible sampling
        
    Returns:
    --------
    X_bootstrap : array-like, shape (n_samples, n_features)
        Bootstrap sample of features
    y_bootstrap : array-like, shape (n_samples,)
        Bootstrap sample of targets
    """
    n = X.shape[0]
    # Generate random indices with replacement
    idx = rng.randint(0, n, size=n)
    return X[idx], y[idx]


class RandomForestClassifierScratch:
    """
    Random Forest Classifier implemented from scratch.
    
    This implementation builds an ensemble of decision trees for classification tasks.
    Each tree is trained on a bootstrap sample of the data and uses random feature
    subsampling. Final predictions are made by majority voting.
    
    How Random Forest Works:
    1. Create multiple bootstrap samples from the training data
    2. Train a decision tree on each bootstrap sample
    3. Each tree uses random feature subsampling for splits
    4. Make predictions by majority voting across all trees
    
    Ensemble Benefits:
    - Reduces variance (overfitting) compared to single trees
    - Improves generalization performance
    - Provides natural regularization through diversity
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
        More trees = better performance but slower training/prediction
        Typically 100-500 trees work well
    
    max_depth : int, default=None
        Maximum depth of each tree
        None means no limit (trees grow until stopping criteria)
        Shorter trees reduce overfitting
    
    max_features : int, default=None
        Number of features to consider when looking for the best split
        None means use all features
        sqrt(n_features) is a common choice for classification
        Reduces correlation between trees
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
        Higher values create more regularized trees
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
        Higher values create more regularized trees
    
    random_state : int, default=None
        Random seed for reproducible results
        Controls both bootstrap sampling and feature subsampling
    """
    
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        # Store hyperparameters
        self.n_estimators = n_estimators            # Number of trees in forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.max_features = max_features            # Max features per split
        self.min_samples_split = min_samples_split  # Min samples to split
        self.min_samples_leaf = min_samples_leaf    # Min samples in leaf
        
        # Initialize random number generator
        self.random_state = np.random.RandomState(random_state)
        
        # Model attributes (set during training)
        self.trees_ = []                            # List of trained trees

    def fit(self, X, y):
        """
        Build a random forest classifier from the training set.
        
        Training Process:
        1. Validate and prepare input data
        2. For each tree in the forest:
           a. Create bootstrap sample from training data
           b. Train decision tree on bootstrap sample
           c. Use random feature subsampling for diversity
        3. Store all trained trees
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training class labels
            
        Returns:
        --------
        self : RandomForestClassifierScratch
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        
        # Initialize list to store trained trees
        self.trees_ = []
        
        # Train each tree in the forest
        for i in range(self.n_estimators):
            # Create bootstrap sample for this tree
            Xi, yi = _bootstrap_sample(X, y, self.random_state)
            
            # Create and train decision tree
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                # Use different random seed for each tree to ensure diversity
                random_state=self.random_state.randint(0, 1_000_000),
            )
            
            # Train the tree on bootstrap sample
            tree.fit(Xi, yi)
            
            # Add trained tree to forest
            self.trees_.append(tree)
        
        return self

    def predict(self, X):
        """
        Predict class labels for the input samples using majority voting.
        
        For each sample, collect predictions from all trees and return the
        class that appears most frequently (majority vote).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Convert input to numpy array
        X = np.asarray(X, dtype=float)
        
        # Get predictions from all trees
        # Each column contains predictions from one tree
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        
        # Perform majority voting for each sample
        final = []
        for row in preds:
            # Count occurrences of each class
            vals, counts = np.unique(row, return_counts=True)
            # Return the class with highest count (majority vote)
            final.append(vals[np.argmax(counts)])
        
        return np.array(final)


class RandomForestRegressorScratch:
    """
    Random Forest Regressor implemented from scratch.
    
    This implementation builds an ensemble of decision trees for regression tasks.
    Each tree is trained on a bootstrap sample of the data and uses random feature
    subsampling. Final predictions are made by averaging predictions from all trees.
    
    How Random Forest Regression Works:
    1. Create multiple bootstrap samples from the training data
    2. Train a decision tree on each bootstrap sample
    3. Each tree uses random feature subsampling for splits
    4. Make predictions by averaging predictions from all trees
    
    Ensemble Benefits:
    - Reduces variance compared to single trees
    - Improves generalization performance
    - Provides natural regularization through diversity
    - Handles non-linear relationships well
    
    Parameters:
    -----------
    n_estimators : int, default=100
        Number of trees in the forest
        More trees = better performance but slower training/prediction
        Typically 100-500 trees work well
    
    max_depth : int, default=None
        Maximum depth of each tree
        None means no limit (trees grow until stopping criteria)
        Shorter trees reduce overfitting
    
    max_features : int, default=None
        Number of features to consider when looking for the best split
        None means use all features
        n_features/3 is a common choice for regression
        Reduces correlation between trees
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
        Higher values create more regularized trees
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
        Higher values create more regularized trees
    
    random_state : int, default=None
        Random seed for reproducible results
        Controls both bootstrap sampling and feature subsampling
    """
    
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        # Store hyperparameters
        self.n_estimators = n_estimators            # Number of trees in forest
        self.max_depth = max_depth                  # Maximum depth of each tree
        self.max_features = max_features            # Max features per split
        self.min_samples_split = min_samples_split  # Min samples to split
        self.min_samples_leaf = min_samples_leaf    # Min samples in leaf
        
        # Initialize random number generator
        self.random_state = np.random.RandomState(random_state)
        
        # Model attributes (set during training)
        self.trees_ = []                            # List of trained trees

    def fit(self, X, y):
        """
        Build a random forest regressor from the training set.
        
        Training Process:
        1. Validate and prepare input data
        2. For each tree in the forest:
           a. Create bootstrap sample from training data
           b. Train decision tree on bootstrap sample
           c. Use random feature subsampling for diversity
        3. Store all trained trees
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training target values
            
        Returns:
        --------
        self : RandomForestRegressorScratch
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Initialize list to store trained trees
        self.trees_ = []
        
        # Train each tree in the forest
        for i in range(self.n_estimators):
            # Create bootstrap sample for this tree
            Xi, yi = _bootstrap_sample(X, y, self.random_state)
            
            # Create and train decision tree
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                # Use different random seed for each tree to ensure diversity
                random_state=self.random_state.randint(0, 1_000_000),
            )
            
            # Train the tree on bootstrap sample
            tree.fit(Xi, yi)
            
            # Add trained tree to forest
            self.trees_.append(tree)
        
        return self

    def predict(self, X):
        """
        Predict target values for the input samples using averaging.
        
        For each sample, collect predictions from all trees and return the
        average of all predictions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted target values
        """
        # Convert input to numpy array
        X = np.asarray(X, dtype=float)
        
        # Get predictions from all trees
        # Each column contains predictions from one tree
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        
        # Average predictions across all trees
        return np.mean(preds, axis=1)