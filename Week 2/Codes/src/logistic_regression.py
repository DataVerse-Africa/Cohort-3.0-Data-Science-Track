import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionGD:
    """
    Binary Logistic Regression trained with Gradient Descent.
    Supports L2 regularization and optional class weights.
    """
    def __init__(self, lr=0.1, n_epochs=2000, l2=0.0, fit_intercept=True, random_state=None):
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

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        Xb = self._add_bias(X)
        n_features = Xb.shape[1]
        w = self.random_state.normal(scale=0.01, size=(n_features, 1))
        if sample_weight is None:
            sample_weight = np.ones((Xb.shape[0], 1))
        else:
            sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1, 1)

        for _ in range(self.n_epochs):
            z = Xb @ w
            p = _sigmoid(z)
            error = (p - y) * sample_weight
            grad = (Xb.T @ error) / Xb.shape[0]
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

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Xb = self._add_bias(X)
        if self.fit_intercept:
            w = np.r_[ [self.intercept_], self.coef_ ].reshape(-1,1)
        else:
            w = self.coef_.reshape(-1,1)
        z = Xb @ w
        p1 = _sigmoid(z).ravel()
        return np.c_[1 - p1, p1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:,1] >= threshold).astype(int)