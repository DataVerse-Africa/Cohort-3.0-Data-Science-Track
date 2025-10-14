"""
Support Vector Machine (SVM) Implementation from Scratch

This module implements a Linear SVM Classifier using Stochastic Gradient Descent (SGD)
optimization on the hinge loss function with L2 regularization.

Support Vector Machines are powerful supervised learning algorithms used for both
classification and regression tasks. The key idea is to find the optimal hyperplane
that separates different classes with the maximum margin.

Key Concepts:
- Maximum Margin: Find the hyperplane that maximizes the distance to the nearest data points
- Support Vectors: Data points that lie closest to the decision boundary
- Hinge Loss: Loss function that penalizes misclassified points and points within the margin
- Soft Margin: Allows some misclassification to handle non-linearly separable data

Mathematical Foundation:
- Objective: minimize (1/2)||w||² + C * Σ max(0, 1 - y_i(w·x_i + b))
- Hinge Loss: L(y, f(x)) = max(0, 1 - y * f(x))
- Decision Function: f(x) = w·x + b
- Prediction: sign(f(x))

Advantages:
- Effective in high-dimensional spaces
- Memory efficient (uses only support vectors)
- Versatile (can use different kernel functions)
- Good generalization performance

Disadvantages:
- Poor performance on large datasets
- Sensitive to feature scaling
- No direct probability estimates
- Can be sensitive to outliers
"""

import numpy as np


class LinearSVMClassifierSGD:
    """
    Linear Support Vector Machine Classifier trained with Stochastic Gradient Descent.
    
    This implementation uses SGD to optimize the hinge loss function with L2 regularization.
    The algorithm finds the optimal hyperplane that separates classes with maximum margin.
    
    How SVM Works:
    1. Initialize weights randomly
    2. For each epoch, shuffle the data
    3. For each sample, compute the margin: y * (w·x + b)
    4. If margin >= 1: sample is correctly classified, only apply regularization
    5. If margin < 1: sample is misclassified or within margin, apply hinge loss gradient
    6. Update weights using gradient descent
    
    Hinge Loss Function:
    - L(y, f(x)) = max(0, 1 - y * f(x))
    - If y * f(x) >= 1: loss = 0 (correctly classified with margin)
    - If y * f(x) < 1: loss = 1 - y * f(x) (penalty for misclassification)
    
    Parameters:
    -----------
    lr : float, default=0.1
        Learning rate for stochastic gradient descent
        Controls the step size in weight updates
        Higher values = faster convergence but may overshoot
    
    n_epochs : int, default=50
        Number of training epochs
        Each epoch processes all training samples once
        More epochs = better convergence but longer training
    
    lam : float, default=1e-3
        L2 regularization strength (inverse of C parameter)
        Higher values = stronger regularization = larger margin
        Lower values = weaker regularization = smaller margin
    
    fit_intercept : bool, default=True
        Whether to fit the intercept term (bias)
        If False, the hyperplane is forced to pass through origin
    
    random_state : int, default=None
        Random seed for reproducible results
        Controls weight initialization and data shuffling
    """
    
    def __init__(self, lr=0.1, n_epochs=50, lam=1e-3, fit_intercept=True, random_state=None):
        # Store hyperparameters
        self.lr = lr                        # Learning rate for SGD
        self.n_epochs = n_epochs            # Number of training epochs
        self.lam = lam                      # L2 regularization strength
        self.fit_intercept = fit_intercept  # Whether to include bias term
        
        # Initialize random number generator
        self.rng = np.random.RandomState(random_state)
        
        # Model parameters (will be learned during training)
        self.w = None                       # Weight vector (includes bias if fit_intercept=True)
        self.intercept_ = 0.0               # Intercept (bias) term
        self.coef_ = None                   # Feature coefficients (weights)

    def _add_bias(self, X):
        """
        Add bias term (intercept) to the feature matrix.
        
        This is done by appending a column of ones to X, which allows us to
        treat the intercept as just another weight in the linear equation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        X_with_bias : array-like, shape (n_samples, n_features+1)
            Features with bias column added
        """
        if not self.fit_intercept:
            return X
        # Add column of ones at the end for bias term
        return np.c_[X, np.ones((X.shape[0], 1))]

    def fit(self, X, y):
        """
        Train the SVM classifier using stochastic gradient descent.
        
        Training Process:
        1. Prepare data (add bias, ensure correct shapes, remap labels)
        2. Initialize weights randomly (small values)
        3. For each epoch:
           a. Shuffle the training data
           b. For each sample:
              - Compute margin: y * (w·x + b)
              - If margin >= 1: only apply regularization gradient
              - If margin < 1: apply hinge loss + regularization gradient
              - Update weights using SGD
        4. Extract final coefficients and intercept
        
        Label Mapping:
        - Input labels can be {0, 1} or {-1, +1}
        - Internally uses {-1, +1} for SVM formulation
        - Maps {0, 1} -> {-1, +1} if needed
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training binary labels (0/1 or -1/+1)
            
        Returns:
        --------
        self : LinearSVMClassifierSGD
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        # Map labels {0,1} -> {-1,+1} if needed
        # SVM formulation requires labels to be {-1, +1}
        if set(np.unique(y)) == {0, 1}:
            y = 2 * y - 1
        
        # Add bias term to feature matrix
        Xb = self._add_bias(X)
        n, d = Xb.shape
        
        # Initialize weights with small random values
        # This helps break symmetry and ensures different starting points
        self.w = self.rng.normal(scale=0.01, size=d)

        # Stochastic Gradient Descent Training Loop
        for epoch in range(self.n_epochs):
            # Shuffle the data for each epoch
            # This helps prevent the algorithm from getting stuck in local minima
            idx = self.rng.permutation(n)
            
            # Process each sample in random order
            for i in idx:
                xi = Xb[i]  # Current sample (with bias)
                yi = y[i]   # Current label
                
                # Compute margin: y * (w·x + b)
                # Positive margin = correctly classified with margin
                # Negative margin = misclassified or within margin
                margin = yi * np.dot(self.w, xi)
                
                # Apply hinge loss and regularization
                if margin >= 1:
                    # Sample is correctly classified with sufficient margin
                    # Only apply L2 regularization gradient
                    if self.fit_intercept:
                        # Don't regularize the bias term (last element)
                        grad = self.lam * np.r_[self.w[:-1], 0]
                    else:
                        # Regularize all weights
                        grad = self.lam * self.w
                else:
                    # Sample is misclassified or within margin
                    # Apply both hinge loss gradient and L2 regularization
                    if self.fit_intercept:
                        # Hinge loss gradient: -y * x (for misclassified samples)
                        # L2 regularization: λ * w (but not for bias term)
                        grad = self.lam * np.r_[self.w[:-1], 0] - yi * xi
                    else:
                        # Apply both gradients to all weights
                        grad = self.lam * self.w - yi * xi
                
                # Update weights using stochastic gradient descent
                # Move in the opposite direction of gradient
                self.w -= self.lr * grad

        # Extract final model parameters
        if self.fit_intercept:
            # If we added bias, separate it from feature coefficients
            self.intercept_ = float(self.w[-1])      # Last weight is the intercept
            self.coef_ = self.w[:-1].copy()          # Remaining weights are coefficients
        else:
            # If no bias was added, all weights are coefficients
            self.coef_ = self.w.copy()
            self.intercept_ = 0.0
            
        return self

    def decision_function(self, X):
        """
        Compute the decision function values for the input samples.
        
        The decision function computes the signed distance from each sample to the
        decision boundary. Positive values indicate class +1, negative values indicate class -1.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns:
        --------
        decision_scores : array-like, shape (n_samples,)
            Decision function values (signed distances to hyperplane)
        """
        # Convert input to numpy array
        X = np.asarray(X, dtype=float)
        
        # Compute decision function: f(x) = w·x + b
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_

    def predict(self, X):
        """
        Predict class labels for the input samples.
        
        This method converts decision function values to binary predictions.
        Samples with decision function >= 0 are predicted as class 1, otherwise class 0.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        # Get decision function scores
        scores = self.decision_function(X)
        
        # Convert scores to binary predictions
        # scores >= 0 -> class 1, scores < 0 -> class 0
        return (scores >= 0).astype(int)