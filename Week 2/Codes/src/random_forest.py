import numpy as np
from .decision_tree import DecisionTreeClassifierScratch, DecisionTreeRegressorScratch

def _bootstrap_sample(X, y, rng):
    n = X.shape[0]
    idx = rng.randint(0, n, size=n)
    return X[idx], y[idx]

class RandomForestClassifierScratch:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = np.random.RandomState(random_state)
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.trees_ = []
        for _ in range(self.n_estimators):
            Xi, yi = _bootstrap_sample(X, y, self.random_state)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state.randint(0, 1_000_000),
            )
            tree.fit(Xi, yi)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        final = []
        for row in preds:
            vals, counts = np.unique(row, return_counts=True)
            final.append(vals[np.argmax(counts)])
        return np.array(final)

class RandomForestRegressorScratch:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = np.random.RandomState(random_state)
        self.trees_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.trees_ = []
        for _ in range(self.n_estimators):
            Xi, yi = _bootstrap_sample(X, y, self.random_state)
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state.randint(0, 1_000_000),
            )
            tree.fit(Xi, yi)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        preds = np.column_stack([t.predict(X) for t in self.trees_])
        return np.mean(preds, axis=1)