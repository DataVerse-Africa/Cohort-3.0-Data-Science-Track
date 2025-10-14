import numpy as np

class _TreeNode:
    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")
    def __init__(self, is_leaf=False, value=None, feature=None, threshold=None, left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def _gini(y):
    m = len(y)
    if m == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / m
    return 1.0 - np.sum(p**2)

def _mse(y):
    if len(y) == 0: return 0.0
    return np.mean((y - np.mean(y))**2)

class DecisionTreeClassifierScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # int or None
        self.random_state = np.random.RandomState(random_state)
        self.n_classes_ = None
        self.tree_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.n_classes_ = len(np.unique(y))
        n_features_total = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features_total
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None
        feat_idx = self.random_state.choice(n, self.max_features, replace=False)
        best_gini = 1.0
        best_feat, best_thr = None, None
        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            for thr in thresholds:
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                g = (left_mask.sum()*_gini(y[left_mask]) + right_mask.sum()*_gini(y[right_mask])) / m
                if g < best_gini:
                    best_gini = g
                    best_feat = f
                    best_thr = thr
        return best_feat, best_thr

    def _leaf_value(self, y):
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def _grow_tree(self, X, y, depth):
        node = _TreeNode()
        if depth == self.max_depth or len(np.unique(y)) == 1 or X.shape[0] < self.min_samples_split:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        feat, thr = self._best_split(X, y)
        if feat is None:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        node.feature, node.threshold = feat, thr
        left_mask = X[:, feat] <= thr
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth+1)
        return node

    def _predict_row(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(x) for x in X])

class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = np.random.RandomState(random_state)
        self.tree_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features_total = X.shape[1]
        if self.max_features is None:
            self.max_features = n_features_total
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None
        feat_idx = self.random_state.choice(n, self.max_features, replace=False)
        best_score = np.inf
        best_feat, best_thr = None, None
        for f in feat_idx:
            thresholds = np.unique(X[:, f])
            for thr in thresholds:
                left_mask = X[:, f] <= thr
                right_mask = ~left_mask
                if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
                    continue
                score = (left_mask.sum()*np.var(y[left_mask]) + right_mask.sum()*np.var(y[right_mask])) / m
                if score < best_score:
                    best_score = score
                    best_feat = f
                    best_thr = thr
        return best_feat, best_thr

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _grow_tree(self, X, y, depth):
        node = _TreeNode()
        if depth == self.max_depth or X.shape[0] < self.min_samples_split:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        feat, thr = self._best_split(X, y)
        if feat is None:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        node.feature, node.threshold = feat, thr
        left_mask = X[:, feat] <= thr
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth+1)
        return node

    def _predict_row(self, x):
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(x) for x in X])