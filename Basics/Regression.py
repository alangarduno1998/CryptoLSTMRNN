import numpy as np
from sklearn import datasets
from collections import Counter
iris = datasets.load_iris()
class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
class PolynomialRegression:
    def __init__(self, lr=0.001, n_iters=1000, degrees=2):
        self.lr = lr
        self.n_iters = n_iters
        self.degrees = degrees
        self.weights = None

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.weights = np.zeros(self.degrees + 1)
        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)

        for _ in range(self.n_iters):
            y_predicted = self.predict(self.X)
            dw = np.dot(X_normalize.T, (y_predicted - self.y))

            self.weights -= self.lr * (1/self.n_samples) * dw
        return self

    def predict(self, X):
        X_transform = self.transform(self.X)
        X_normalize = self.normalize(X_transform)
        y_predicted = np.dot(X, self.weights)
        return y_predicted
    def transform(self, X):
        X_transform = np.ones((self.n_samples,1))
        for j in range(self.degree + 1):
            if j != 0:
                x_pow = np.power(X,j)
                X_transform = np.append(X_transform, x_pow.reshape(-1,1), axis = 1)
        return X_transform

    def normalize(self, X):
        X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
        return X

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            linear_model = np.clip(linear_model, -709.78, 709.78)
            y_predicted = self._sigmoid(linear_model)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        linear_model = np.clip(linear_model, -709.78, 709.78)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape() # rows and columns
        n_classes= np.unique(y)
        self._classes= len(n_classes)
        self.weights=np.zeros(n_features)
        self.bias = 0

        #init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for c in n_classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0]/ float(n_samples)


    def predict(self, X, y):
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors(idx))
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posterior = prior + class_conditional
            posteriors.append(posteriors)

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator
class Perceptron:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X,y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_= np.array([1 if i>0.5 else 0 for i in y])

        # perceptron
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_predicted =self.activation_func(linear_model)
                dw = self.lr * (y_[idx]-y_predicted)

            self.weights += dw * x_i
            self.bias += dw


    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._unit_step_func(linear_model)
        return y_predicted

    def _unit_step_func(self,x):
        return np.where(x>=0,1,0)
class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w= None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] *(np.dot(x_i, self.w)-self.b)>=1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_model = np.dat(X, self.w) - self.b
        return np.sign(linear_model)
class PCA:

    def __init__(self, n_components):

        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        #mean
        self.mean = np.mean(X, axis=0)
        # need mean subtraction so first principal component describes direction of max variance
        X = X - self.mean
        #covariance
        # row = 1 sample, columns = feature
        cov = np.cov(X.T)

        #calculate eigen vecotersa and eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        #v[:, i]
        #sort eigen vectors, by eigen values
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        #store first n eignevgectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X- self.mean
        return np.dot(X, self.components.T)
def entropy(y):
    hist = np.bincount(y)
    ps= hist/len(y)
    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node:
    def __init__(self, feature=None, threshold=None, left =None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # grow tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self,X,y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth or n_labels ==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace = False)

        #greedy search
        best_feat, best_thresh = self._best_criteria(X,y,feat_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    def _best_criteria(self,X,y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        #parent E
        parent_entropy = entropy(y)
        #generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs)==0 or len(right_idxs) == 0:
            return 0

        #weighted avg child E
        n=len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l +(n_r/n)* e_r

        #return ig
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self,y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        #traverse tree
        return np.array([self._traverse_tree(x,self.root) for x in X])
    def _traverse_tree(self,x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples,size=n_samples, replace=True)
    return X[idxs], y[idxs]
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common
class RandomForest:
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats

    def fit(self, X,y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # [1111 0000 1111]
        # [101 101 101 101]
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)



