"""
Linear Regression Implementation from Scratch

This module contains two implementations of Linear Regression:
1. LinearRegressionGD: Uses Gradient Descent optimization
2. LinearRegressionNormal: Uses the Normal Equation (closed-form solution)

Linear Regression is a fundamental machine learning algorithm that models the relationship
between a dependent variable (y) and one or more independent variables (X) using a linear equation:
y = w0 + w1*x1 + w2*x2 + ... + wn*xn

Where:
- w0 is the intercept (bias term)
- w1, w2, ..., wn are the coefficients (weights) for each feature
"""

import numpy as np


class LinearRegressionGD:
    """
    Linear Regression trained with (batch) Gradient Descent.
    
    This implementation uses the iterative gradient descent algorithm to find the optimal
    weights that minimize the Mean Squared Error (MSE) loss function.
    
    Mathematical Foundation:
    - Loss function: MSE = (1/2m) * Σ(y_pred - y_true)²
    - Gradient: ∇w = (1/m) * X^T * (X*w - y)
    - Update rule: w = w - learning_rate * gradient
    
    Advantages of Gradient Descent:
    - Works well with large datasets (memory efficient)
    - Can handle online learning (incremental updates)
    - Supports regularization easily
    
    Parameters:
    -----------
    lr : float, default=0.01
        Learning rate - controls the step size in gradient descent
        Too high: may overshoot optimal solution
        Too low: slow convergence
    
    n_epochs : int, default=1000
        Number of iterations to run gradient descent
        More epochs = better convergence but longer training time
    
    l2 : float, default=0.0
        L2 regularization strength (Ridge regression)
        Helps prevent overfitting by penalizing large weights
        Higher values = stronger regularization
    
    fit_intercept : bool, default=True
        Whether to fit the intercept term (bias)
        If False, the line is forced to pass through origin
    
    random_state : int, default=None
        Random seed for reproducible results
    """
    
    def __init__(self, lr=0.01, n_epochs=1000, l2=0.0, fit_intercept=True, random_state=None):
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
        
        Before: y = w1*x1 + w2*x2 + ... + wn*xn
        After:  y = w0*1 + w1*x1 + w2*x2 + ... + wn*xn
        
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

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Training Process:
        1. Prepare data (add bias, ensure correct shapes)
        2. Initialize weights randomly (small values)
        3. For each epoch:
           a. Compute predictions: y_pred = X * w
           b. Compute error: error = y_pred - y_true
           c. Compute gradient: grad = (1/m) * X^T * error
           d. Add L2 regularization to gradient
           e. Update weights: w = w - lr * gradient
        4. Extract final coefficients and intercept
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : LinearRegressionGD
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

        # Gradient Descent Training Loop
        for epoch in range(self.n_epochs):
            # Step 1: Compute predictions using current weights
            # Matrix multiplication: Xb @ w gives predictions for all samples
            y_pred = Xb @ w
            
            # Step 2: Compute prediction errors
            # This is the difference between predicted and actual values
            error = y_pred - y
            
            # Step 3: Compute gradient of the loss function
            # Gradient = (1/m) * X^T * error
            # This tells us the direction and magnitude of steepest increase in loss
            grad = (Xb.T @ error) / Xb.shape[0]
            
            # Step 4: Add L2 regularization to prevent overfitting
            # L2 penalty: λ * w (but don't regularize the bias term)
            if self.l2 > 0:
                # Create regularization term: [0, λ*w1, λ*w2, ..., λ*wn]
                # First element is 0 to avoid regularizing bias term
                reg = self.l2 * np.r_[np.zeros((1,1)), w[1:]]
                grad += reg
            
            # Step 5: Update weights using gradient descent
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

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        The prediction formula is: y_pred = X * coef_ + intercept_
        
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
        
        # Apply the linear equation: y = X * w + b
        # X @ self.coef_.reshape(-1,) computes the dot product for each sample
        # Then add the intercept term
        return (X @ self.coef_.reshape(-1,)) + self.intercept_


class LinearRegressionNormal:
    """
    Linear Regression solved by the Normal Equation (closed-form solution).
    
    This implementation uses the mathematical closed-form solution to find the optimal
    weights directly, without iterative optimization.
    
    Mathematical Foundation:
    - Normal Equation: w = (X^T * X)^(-1) * X^T * y
    - This is derived by setting the gradient of MSE to zero
    - Uses pseudo-inverse (pinv) for numerical stability
    
    Advantages of Normal Equation:
    - No need to choose learning rate
    - No iterations required (single computation)
    - Guaranteed to find global minimum (for linear problems)
    
    Disadvantages:
    - Computationally expensive for large datasets: O(n³) complexity
    - Requires matrix inversion (may be unstable for singular matrices)
    - Memory intensive for large feature matrices
    
    When to use Normal Equation vs Gradient Descent:
    - Normal Equation: n_features < 10,000 and well-conditioned matrices
    - Gradient Descent: large datasets or high-dimensional features
    
    Parameters:
    -----------
    fit_intercept : bool, default=True
        Whether to fit the intercept term (bias)
        If False, the line is forced to pass through origin
    """
    
    def __init__(self, fit_intercept=True):
        # Store hyperparameters
        self.fit_intercept = fit_intercept  # Whether to include bias term
        
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

    def fit(self, X, y):
        """
        Train the linear regression model using the Normal Equation.
        
        Training Process:
        1. Prepare data (add bias, ensure correct shapes)
        2. Apply Normal Equation: w = (X^T * X)^(-1) * X^T * y
        3. Use pseudo-inverse for numerical stability
        4. Extract coefficients and intercept
        
        The Normal Equation is derived as follows:
        - Start with MSE loss: L = (1/2m) * ||X*w - y||²
        - Take gradient and set to zero: ∇L = (1/m) * X^T * (X*w - y) = 0
        - Solve for w: X^T * X * w = X^T * y
        - Therefore: w = (X^T * X)^(-1) * X^T * y
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training targets
            
        Returns:
        --------
        self : LinearRegressionNormal
            Returns self for method chaining
        """
        # Convert inputs to numpy arrays and ensure correct data types
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)  # Reshape to column vector
        
        # Add bias term to feature matrix
        Xb = self._add_bias(X)
        
        # Apply the Normal Equation: w = (X^T * X)^(-1) * X^T * y
        # We use pseudo-inverse (pinv) instead of regular inverse for numerical stability
        # pinv handles cases where X^T * X is singular or near-singular
        w = np.linalg.pinv(Xb) @ y
        
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

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        The prediction formula is: y_pred = X * coef_ + intercept_
        
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
        
        # Apply the linear equation: y = X * w + b
        # X @ self.coef_.reshape(-1,) computes the dot product for each sample
        # Then add the intercept term
        return (X @ self.coef_.reshape(-1,)) + self.intercept_