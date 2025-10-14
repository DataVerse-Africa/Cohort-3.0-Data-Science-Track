import numpy as np

class LinearSVMClassifierSGD:
    """
    Linear soft-margin SVM trained with SGD on hinge loss + L2 regularization.
    Labels y must be in {-1, +1} (0/1 will be remapped).
    """
    def __init__(self, lr=0.1, n_epochs=50, lam=1e-3, fit_intercept=True, random_state=None):
        self.lr = lr
        self.n_epochs = n_epochs
        self.lam = lam
        self.fit_intercept = fit_intercept
        self.rng = np.random.RandomState(random_state)
        self.w = None
        self.intercept_ = 0.0
        self.coef_ = None

    def _add_bias(self, X):
        if not self.fit_intercept:
            return X
        return np.c_[X, np.ones((X.shape[0], 1))]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        # map labels {0,1} -> {-1,+1} if needed
        if set(np.unique(y)) == {0,1}:
            y = 2*y - 1
        Xb = self._add_bias(X)
        n, d = Xb.shape
        self.w = self.rng.normal(scale=0.01, size=d)

        for _ in range(self.n_epochs):
            idx = self.rng.permutation(n)
            for i in idx:
                xi = Xb[i]
                yi = y[i]
                margin = yi * np.dot(self.w, xi)
                if margin >= 1:
                    # only regularization gradient (do not reg bias term)
                    grad = self.lam * np.r_[self.w[:-1], 0] if self.fit_intercept else self.lam * self.w
                else:
                    grad = (self.lam * np.r_[self.w[:-1], 0] - yi * xi) if self.fit_intercept else (self.lam * self.w - yi * xi)
                self.w -= self.lr * grad

        if self.fit_intercept:
            self.intercept_ = float(self.w[-1])
            self.coef_ = self.w[:-1].copy()
        else:
            self.coef_ = self.w.copy()
            self.intercept_ = 0.0
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)