"""
Decision Tree Implementation from Scratch

This module implements both Classification and Regression Decision Trees from scratch.

Decision Trees are non-parametric supervised learning algorithms that can be used for both
classification and regression tasks. They work by recursively partitioning the feature space
into regions and making predictions based on the majority class (classification) or mean value
(regression) in each region.

Key Concepts:
- Tree Structure: Each node represents a decision based on a feature and threshold
- Splitting Criterion: Gini Impurity (classification) or MSE (regression)
- Recursive Partitioning: Continue splitting until stopping criteria are met
- Leaf Nodes: Make final predictions

Advantages:
- Easy to understand and interpret
- Can handle both numerical and categorical data
- Requires little data preparation
- Can model non-linear relationships

Disadvantages:
- Prone to overfitting
- Can be unstable (small changes in data can lead to different trees)
- Biased towards features with more levels
"""

import numpy as np


class _TreeNode:
    """
    Internal node structure for the decision tree.
    
    This class represents a single node in the decision tree. Each node can be either:
    - Internal node: Contains a decision rule (feature + threshold) and child nodes
    - Leaf node: Contains the final prediction value
    
    Attributes:
    -----------
    is_leaf : bool
        True if this is a leaf node (terminal node), False if internal node
    value : float or int
        Prediction value (only used for leaf nodes)
    feature : int
        Index of the feature used for splitting (only used for internal nodes)
    threshold : float
        Threshold value for the split (only used for internal nodes)
    left : _TreeNode
        Left child node (samples <= threshold)
    right : _TreeNode
        Right child node (samples > threshold)
    """
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")
    
    def __init__(self, is_leaf=False, value=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf          # Whether this is a terminal (leaf) node
        self.value = value              # Prediction value (for leaf nodes)
        self.feature = feature          # Feature index for splitting (for internal nodes)
        self.threshold = threshold      # Threshold for splitting (for internal nodes)
        self.left = left                # Left child node (≤ threshold)
        self.right = right              # Right child node (> threshold)


def _gini(y):
    """
    Calculate Gini Impurity for classification.
    
    Gini Impurity measures how often a randomly chosen element would be incorrectly
    labeled if it was randomly labeled according to the distribution of labels in the subset.
    
    Mathematical Formula:
    Gini = 1 - Σ(p_i²) where p_i is the proportion of class i
    
    Properties:
    - Range: [0, 0.5] for binary classification, [0, 1-1/k] for k classes
    - Gini = 0: perfect purity (all samples belong to same class)
    - Gini = 0.5: maximum impurity (equal distribution of classes)
    
    Parameters:
    -----------
    y : array-like
        Array of class labels
        
    Returns:
    --------
    gini : float
        Gini impurity value
    """
    m = len(y)
    if m == 0: 
        return 0.0
    
    # Count occurrences of each unique class
    _, counts = np.unique(y, return_counts=True)
    
    # Calculate proportions (probabilities) of each class
    p = counts / m
    
    # Calculate Gini impurity: 1 - Σ(p_i²)
    return 1.0 - np.sum(p**2)


def _mse(y):
    """
    Calculate Mean Squared Error for regression.
    
    MSE measures the average squared difference between the actual values and the mean.
    This is used as the impurity measure for regression trees.
    
    Mathematical Formula:
    MSE = (1/n) * Σ(y_i - ȳ)² where ȳ is the mean of y
    
    Properties:
    - Range: [0, +∞)
    - MSE = 0: perfect purity (all samples have same value)
    - Higher MSE: more variance in the target values
    
    Parameters:
    -----------
    y : array-like
        Array of target values
        
    Returns:
    --------
    mse : float
        Mean squared error value
    """
    if len(y) == 0: 
        return 0.0
    
    # Calculate mean of target values
    mean_y = np.mean(y)
    
    # Calculate MSE: average of squared differences from mean
    return np.mean((y - mean_y)**2)

class DecisionTreeClassifierScratch:
    """
    Decision Tree Classifier implemented from scratch.
    
    This implementation builds a binary decision tree for classification tasks using
    the Gini impurity criterion to determine the best splits.
    
    How Decision Trees Work:
    1. Start with all training data at the root node
    2. Find the best feature and threshold to split the data
    3. Split the data into left and right child nodes
    4. Recursively repeat for each child node
    5. Stop when stopping criteria are met (leaf nodes)
    6. Make predictions by traversing the tree from root to leaf
    
    Splitting Criterion:
    - Uses Gini Impurity to measure the "purity" of a split
    - Gini = 1 - Σ(p_i²) where p_i is the proportion of class i
    - Lower Gini = more pure split (better separation of classes)
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree
        None means no limit (tree grows until stopping criteria)
        Prevents overfitting by limiting tree complexity
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
        Higher values prevent overfitting by requiring more samples for splits
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
        Higher values create more regularized trees
    
    max_features : int, default=None
        Number of features to consider when looking for the best split
        None means use all features
        Used for feature subsampling (similar to Random Forest)
    
    random_state : int, default=None
        Random seed for reproducible results
        Used for feature subsampling when max_features < n_features
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        # Store hyperparameters
        self.max_depth = max_depth                    # Maximum tree depth
        self.min_samples_split = min_samples_split    # Min samples to split
        self.min_samples_leaf = min_samples_leaf      # Min samples in leaf
        self.max_features = max_features              # Max features to consider
        
        # Initialize random number generator
        self.random_state = np.random.RandomState(random_state)
        
        # Model attributes (set during training)
        self.n_classes_ = None                        # Number of classes
        self.tree_ = None                             # Root node of the tree

    def fit(self, X, y):
        """
        Build a decision tree classifier from the training set.
        
        Training Process:
        1. Validate and prepare input data
        2. Determine number of classes
        3. Set up feature sampling parameters
        4. Grow the tree recursively starting from root
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training class labels
            
        Returns:
        --------
        self : DecisionTreeClassifierScratch
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        
        # Store number of classes
        self.n_classes_ = len(np.unique(y))
        
        # Set up feature sampling
        n_features_total = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features_total
        
        # Grow the tree starting from root
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.
        
        This method tries all possible splits and returns the one that minimizes
        the weighted Gini impurity of the resulting child nodes.
        
        Split Quality Calculation:
        - For each feature and threshold, calculate weighted Gini impurity
        - Weighted Gini = (n_left/n_total) * Gini_left + (n_right/n_total) * Gini_right
        - Choose the split that minimizes this weighted impurity
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target labels
            
        Returns:
        --------
        best_feature : int or None
            Index of the best feature to split on
        best_threshold : float or None
            Best threshold value for the split
        """
        m, n = X.shape
        
        # Check if we have enough samples to split
        if m < self.min_samples_split:
            return None, None
        
        # Randomly select features to consider (for feature subsampling)
        feat_idx = self.random_state.choice(n, self.max_features, replace=False)
        
        # Initialize variables to track the best split
        best_gini = 1.0  # Start with worst possible Gini (maximum impurity)
        best_feat, best_thr = None, None
        
        # Try each selected feature
        for f in feat_idx:
            # Get unique values of this feature as potential thresholds
            thresholds = np.unique(X[:, f])
            
            # Try each threshold
            for thr in thresholds:
                # Create masks for left and right splits
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                
                # Check if both splits have enough samples
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                
                # Calculate weighted Gini impurity for this split
                # Weighted Gini = (n_left/n_total) * Gini_left + (n_right/n_total) * Gini_right
                g = (left_mask.sum() * _gini(y[left_mask]) + 
                     right_mask.sum() * _gini(y[right_mask])) / m
                
                # Update best split if this one is better
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thr = thr
        
        return best_feat, best_thr

    def _leaf_value(self, y):
        """
        Determine the prediction value for a leaf node.
        
        For classification, this returns the majority class (most frequent class)
        in the leaf node.
        
        Parameters:
        -----------
        y : array-like
            Target labels in the leaf node
            
        Returns:
        --------
        prediction : int
            Majority class label
        """
        # Count occurrences of each class
        vals, counts = np.unique(y, return_counts=True)
        
        # Return the class with the highest count (majority class)
        return int(vals[np.argmax(counts)])

    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree.
        
        This is the core recursive function that builds the tree by:
        1. Checking stopping criteria
        2. Finding the best split
        3. Creating child nodes and recursively growing them
        
        Stopping Criteria:
        - Maximum depth reached
        - All samples belong to the same class (perfect purity)
        - Not enough samples to split
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for current node
        y : array-like, shape (n_samples,)
            Target labels for current node
        depth : int
            Current depth in the tree
            
        Returns:
        --------
        node : _TreeNode
            Root node of the subtree
        """
        # Create a new node
        node = _TreeNode()
        
        # Check stopping criteria
        if (depth == self.max_depth or                    # Max depth reached
            len(np.unique(y)) == 1 or                     # All samples same class
            X.shape[0] < self.min_samples_split):         # Not enough samples to split
            
            # Create leaf node
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Find the best split
        feat, thr = self._best_split(X, y)
        
        # If no good split found, create leaf node
        if feat is None:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Create internal node with the best split
        node.feature, node.threshold = feat, thr
        
        # Split the data
        left_mask = X[:, feat] <= thr
        
        # Recursively grow left and right subtrees
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return node

    def _predict_row(self, x):
        """
        Predict the class for a single sample by traversing the tree.
        
        Starting from the root, follow the decision path based on feature values
        until reaching a leaf node, then return the leaf's prediction.
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Single sample to predict
            
        Returns:
        --------
        prediction : int
            Predicted class label
        """
        node = self.tree_
        
        # Traverse the tree until we reach a leaf
        while not node.is_leaf:
            # Go left if feature value <= threshold, right otherwise
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # Return the prediction from the leaf node
        return node.value

    def predict(self, X):
        """
        Predict class labels for the input samples.
        
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
        
        # Predict each sample by traversing the tree
        return np.array([self._predict_row(x) for x in X])

class DecisionTreeRegressorScratch:
    """
    Decision Tree Regressor implemented from scratch.
    
    This implementation builds a binary decision tree for regression tasks using
    the Mean Squared Error (MSE) criterion to determine the best splits.
    
    How Regression Trees Work:
    1. Start with all training data at the root node
    2. Find the best feature and threshold to split the data
    3. Split the data into left and right child nodes
    4. Recursively repeat for each child node
    5. Stop when stopping criteria are met (leaf nodes)
    6. Make predictions by traversing the tree from root to leaf
    
    Splitting Criterion:
    - Uses variance reduction to measure the quality of a split
    - For regression, we minimize the weighted variance of child nodes
    - Lower variance = more homogeneous values in each child node
    
    Key Differences from Classification Tree:
    - Uses variance instead of Gini impurity
    - Leaf nodes predict the mean value instead of majority class
    - Target values are continuous instead of discrete classes
    
    Parameters:
    -----------
    max_depth : int, default=None
        Maximum depth of the tree
        None means no limit (tree grows until stopping criteria)
        Prevents overfitting by limiting tree complexity
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node
        Higher values prevent overfitting by requiring more samples for splits
    
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node
        Higher values create more regularized trees
    
    max_features : int, default=None
        Number of features to consider when looking for the best split
        None means use all features
        Used for feature subsampling (similar to Random Forest)
    
    random_state : int, default=None
        Random seed for reproducible results
        Used for feature subsampling when max_features < n_features
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        # Store hyperparameters
        self.max_depth = max_depth                    # Maximum tree depth
        self.min_samples_split = min_samples_split    # Min samples to split
        self.min_samples_leaf = min_samples_leaf      # Min samples in leaf
        self.max_features = max_features              # Max features to consider
        
        # Initialize random number generator
        self.random_state = np.random.RandomState(random_state)
        
        # Model attributes (set during training)
        self.tree_ = None                             # Root node of the tree

    def fit(self, X, y):
        """
        Build a decision tree regressor from the training set.
        
        Training Process:
        1. Validate and prepare input data
        2. Set up feature sampling parameters
        3. Grow the tree recursively starting from root
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training target values
            
        Returns:
        --------
        self : DecisionTreeRegressorScratch
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Set up feature sampling
        n_features_total = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features_total
        
        # Grow the tree starting from root
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.
        
        This method tries all possible splits and returns the one that minimizes
        the weighted variance of the resulting child nodes.
        
        Split Quality Calculation:
        - For each feature and threshold, calculate weighted variance
        - Weighted Variance = (n_left/n_total) * Var_left + (n_right/n_total) * Var_right
        - Choose the split that minimizes this weighted variance
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        best_feature : int or None
            Index of the best feature to split on
        best_threshold : float or None
            Best threshold value for the split
        """
        m, n = X.shape
        
        # Check if we have enough samples to split
        if m < self.min_samples_split:
            return None, None
        
        # Randomly select features to consider (for feature subsampling)
        feat_idx = self.random_state.choice(n, self.max_features, replace=False)
        
        # Initialize variables to track the best split
        best_score = np.inf  # Start with worst possible score (infinite variance)
        best_feat, best_thr = None, None
        
        # Try each selected feature
        for f in feat_idx:
            # Get unique values of this feature as potential thresholds
            thresholds = np.unique(X[:, f])
            
            # Try each threshold
            for thr in thresholds:
                # Create masks for left and right splits
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                
                # Check if both splits have enough samples
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                
                # Calculate weighted variance for this split
                # Weighted Variance = (n_left/n_total) * Var_left + (n_right/n_total) * Var_right
                score = (left_mask.sum() * np.var(y[left_mask]) + 
                        right_mask.sum() * np.var(y[right_mask])) / m
                
                # Update best split if this one is better
                if score < best_score:
                    best_score = score
                    best_feat = f
                    best_thr = thr
        
        return best_feat, best_thr

    def _leaf_value(self, y):
        """
        Determine the prediction value for a leaf node.
        
        For regression, this returns the mean value of all target values
        in the leaf node.
        
        Parameters:
        -----------
        y : array-like
            Target values in the leaf node
            
        Returns:
        --------
        prediction : float
            Mean of target values
        """
        return float(np.mean(y))

    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree.
        
        This is the core recursive function that builds the tree by:
        1. Checking stopping criteria
        2. Finding the best split
        3. Creating child nodes and recursively growing them
        
        Stopping Criteria:
        - Maximum depth reached
        - Not enough samples to split
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for current node
        y : array-like, shape (n_samples,)
            Target values for current node
        depth : int
            Current depth in the tree
            
        Returns:
        --------
        node : _TreeNode
            Root node of the subtree
        """
        # Create a new node
        node = _TreeNode()
        
        # Check stopping criteria
        if (depth == self.max_depth or                    # Max depth reached
            X.shape[0] < self.min_samples_split):         # Not enough samples to split
            
            # Create leaf node
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Find the best split
        feat, thr = self._best_split(X, y)
        
        # If no good split found, create leaf node
        if feat is None:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Create internal node with the best split
        node.feature, node.threshold = feat, thr
        
        # Split the data
        left_mask = X[:, feat] <= thr
        
        # Recursively grow left and right subtrees
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return node

    def _predict_row(self, x):
        """
        Predict the target value for a single sample by traversing the tree.
        
        Starting from the root, follow the decision path based on feature values
        until reaching a leaf node, then return the leaf's prediction.
        
        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Single sample to predict
            
        Returns:
        --------
        prediction : float
            Predicted target value
        """
        node = self.tree_
        
        # Traverse the tree until we reach a leaf
        while not node.is_leaf:
            # Go left if feature value <= threshold, right otherwise
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        # Return the prediction from the leaf node
        return node.value

    def predict(self, X):
        """
        Predict target values for the input samples.
        
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
        
        # Predict each sample by traversing the tree
        return np.array([self._predict_row(x) for x in X])