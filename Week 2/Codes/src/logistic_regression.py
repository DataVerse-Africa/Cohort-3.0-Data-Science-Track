"""
Logistic Regression Implementation from Scratch

This module implements Binary Logistic Regression using Gradient Descent optimization.

Logistic Regression is a classification algorithm that models the probability of a binary outcome
using the logistic (sigmoid) function. Unlike linear regression which outputs continuous values,
logistic regression outputs probabilities between 0 and 1.

Mathematical Foundation:
- Hypothesis: h(x) = σ(w^T * x + b) where σ is the sigmoid function
- Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Cost function: J(w) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
- Gradient: ∇J = (1/m) * X^T * (h(x) - y)

The sigmoid function maps any real number to a value between 0 and 1, making it perfect
for probability estimation. The decision boundary is typically set at 0.5.
"""

import numpy as np


def _sigmoid(z):
    """
    Sigmoid (logistic) activation function.
    
    The sigmoid function maps any real number to a value between 0 and 1.
    This makes it perfect for binary classification as it can represent probabilities.
    
    Mathematical properties:
    - Range: (0, 1)
    - S-shaped curve (S-curve)
    - σ(0) = 0.5
    - σ(z) → 1 as z → +∞
    - σ(z) → 0 as z → -∞
    
    Parameters:
    -----------
    z : array-like
        Input values (can be scalar or array)
        
    Returns:
    --------
    sigmoid_values : array-like
        Sigmoid function values, same shape as input
    """
    # Clip z to prevent overflow in exp(-z) for very large negative values
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionGD:
    """
    Binary Logistic Regression trained with Gradient Descent.
    
    This implementation uses the iterative gradient descent algorithm to find the optimal
    weights that minimize the cross-entropy loss function.
    
    Mathematical Foundation:
    - Hypothesis: h(x) = σ(w^T * x + b)
    - Loss function: J(w) = -(1/m) * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
    - Gradient: ∇J = (1/m) * X^T * (h(x) - y)
    - Update rule: w = w - learning_rate * gradient
    
    Key Differences from Linear Regression:
    - Uses sigmoid activation function instead of linear output
    - Minimizes cross-entropy loss instead of MSE
    - Outputs probabilities instead of continuous values
    - Used for classification instead of regression
    
    Parameters:
    -----------
    lr : float, default=0.1
        Learning rate - controls the step size in gradient descent
        Higher learning rate for logistic regression (compared to linear regression)
        because the sigmoid function has bounded gradients
    
    n_epochs : int, default=2000
        Number of iterations to run gradient descent
        Logistic regression typically needs more epochs than linear regression
    
    l2 : float, default=0.0
        L2 regularization strength (Ridge regularization)
        Helps prevent overfitting by penalizing large weights
    
    fit_intercept : bool, default=True
        Whether to fit the intercept term (bias)
        If False, the decision boundary is forced to pass through origin
    
    random_state : int, default=None
        Random seed for reproducible results
    """
    
    def __init__(self, lr=0.1, n_epochs=2000, l2=0.0, fit_intercept=True, random_state=None):
        # Store hyperparameters
        self.lr = lr                    # Learning rate for gradient descent
        self.n_epochs = n_epochs        # Number of training iterations
        self.l2 = l2                    # L2 regularization strength
        self.fit_intercept = fit_intercept  # Whether to include bias term
        
        # Initialize random number generator for reproducible results
        self.random_state = np.random.RandomState(random_state)
        
        # Model parameters (will be learned during training)
        self.coef_ = None               # Feature coefficients (weights)
        self.intercept_ = 0.0           # Intercept (bias) term

    def _add_bias(self, X):
        """
        Add bias term (intercept) to the feature matrix.
        
        This is done by prepending a column of ones to X, which allows us to
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
        # Add column of ones at the beginning for bias term
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y, sample_weight=None):
        """
        Train the logistic regression model using gradient descent.
        
        Training Process:
        1. Prepare data (add bias, ensure correct shapes)
        2. Initialize weights randomly (small values)
        3. For each epoch:
           a. Compute linear combination: z = X * w
           b. Apply sigmoid: p = σ(z)
           c. Compute error: error = p - y
           d. Apply sample weights if provided
           e. Compute gradient: grad = (1/m) * X^T * error
           f. Add L2 regularization to gradient
           g. Update weights: w = w - lr * gradient
        4. Extract final coefficients and intercept
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training binary labels (0 or 1)
        sample_weight : array-like, shape (n_samples,), optional
            Individual weights for each sample
            Useful for handling class imbalance or giving more importance to certain samples
            
        Returns:
        --------
        self : LogisticRegressionGD
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays and ensure correct data types
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)  # Reshape to column vector
        
        # Add bias term to feature matrix
        Xb = self._add_bias(X)
        n_features = Xb.shape[1]
        
        # Initialize weights with small random values
        # This helps break symmetry and ensures different starting points
        w = self.random_state.normal(scale=0.01, size=(n_features, 1))
        
        # Handle sample weights
        if sample_weight is None:
            # If no sample weights provided, use equal weights for all samples
            sample_weight = np.ones((Xb.shape[0], 1))
        else:
            # Convert sample weights to numpy array and reshape
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)

        # Gradient Descent Training Loop
        for epoch in range(self.n_epochs):
            # Step 1: Compute linear combination (logits)
            # This is the input to the sigmoid function
            z = Xb @ w
            
            # Step 2: Apply sigmoid activation to get probabilities
            # This maps logits to probabilities between 0 and 1
            p = _sigmoid(z)
            
            # Step 3: Compute prediction errors
            # This is the difference between predicted probabilities and actual labels
            error = (p - y) * sample_weight
            
            # Step 4: Compute gradient of the cross-entropy loss
            # Gradient = (1/m) * X^T * error
            # This tells us the direction and magnitude of steepest increase in loss
            grad = (Xb.T @ error) / Xb.shape[0]
            
            # Step 5: Add L2 regularization to prevent overfitting
            # L2 penalty: λ * w (but don't regularize the bias term)
            if self.l2 > 0:
                # Create regularization term: [0, λ*w1, λ*w2, ..., λ*wn]
                # First element is 0 to avoid regularizing bias term
                reg = self.l2 * np.r_[np.zeros((1,1)), w[1:]]
                grad += reg
            
            # Step 6: Update weights using gradient descent
            # Move in the opposite direction of gradient (steepest descent)
            w -= self.lr * grad

        # Extract final model parameters
        if self.fit_intercept:
            # If we added bias, separate it from feature coefficients
            self.intercept_ = float(w[0])      # First weight is the intercept
            self.coef_ = w[1:].ravel()         # Remaining weights are coefficients
        else:
            # If no bias was added, all weights are coefficients
            self.coef_ = w.ravel()
            self.intercept_ = 0.0
            
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for the input samples.
        
        This method returns the probability of each class (0 and 1) for each sample.
        The probabilities sum to 1 for each sample.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
            
        Returns:
        --------
        probabilities : array-like, shape (n_samples, 2)
            Class probabilities for each sample
            Column 0: probability of class 0
            Column 1: probability of class 1
        """
        # Convert input to numpy array
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        
        # Reconstruct weight vector
        if self.fit_intercept:
            # Combine intercept and coefficients back into single weight vector
            w = np.r_[ [self.intercept_], self.coef_ ].reshape(-1,1)
        else:
            # Use only coefficients if no intercept
            w = self.coef_.reshape(-1,1)
        
        # Compute linear combination (logits)
        z = Xb @ w
        
        # Apply sigmoid to get probability of class 1
        p1 = _sigmoid(z).ravel()
        
        # Return probabilities for both classes
        # p0 = 1 - p1 (probability of class 0)
        return np.c_[1 - p1, p1]

    def predict(self, X, threshold=0.5):
        """
        Predict class labels for the input samples.
        
        This method converts probabilities to binary predictions using a threshold.
        Samples with probability >= threshold are predicted as class 1, otherwise class 0.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features for prediction
        threshold : float, default=0.5
            Decision threshold for binary classification
            Values >= threshold are predicted as class 1
            
        Returns:
        --------
        predictions : array-like, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        # Get probabilities for class 1 (second column)
        prob_class_1 = self.predict_proba(X)[:, 1]
        
        # Convert probabilities to binary predictions using threshold
        return (prob_class_1 >= threshold).astype(int)