import numpy as np

class LinearRegressionGD:
    """
    Linear Regression trained with (batch) Gradient Descent.
    Supports optional L2 regularization.
    """
    def __init__(self, lr=0.01, n_epochs=1000, l2=0.0, fit_intercept=True, random_state=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.random_state = np.random.RandomState(random_state)
        self.coef_ = None
        self.intercept_ = 0.0

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = self._add_bias(X)
        n_features = Xb.shape[1]
        w = self.random_state.normal(scale=0.01, size=(n_features, 1))

        for _ in range(self.n_epochs):
            y_pred = Xb @ w
            error = y_pred - y
            grad = (Xb.T @ error) / Xb.shape[0]
            # L2 regularization (do not regularize bias term)
            if self.l2 > 0:
                reg = self.l2 * np.r_[np.zeros((1,1)), w[1:]]
                grad += reg
            w -= self.lr * grad

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].ravel()
        else:
            self.coef_ = w.ravel()
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.coef_.reshape(-1,)) + self.intercept_


class LinearRegressionNormal:
    """
    Linear Regression solved by the Normal Equation (uses pseudo-inverse).
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[np.ones((X.shape[0], 1)), X]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = self._add_bias(X)
        # w = (X^T X)^(-1) X^T y  -> use pinv for stability
        w = np.linalg.pinv(Xb) @ y
        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].ravel()
        else:
            self.coef_ = w.ravel()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.coef_.reshape(-1,)) + self.intercept_