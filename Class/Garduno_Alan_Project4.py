import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
class LogisticRegression2:
    def sigmoid(self,z):
        sig = 1/(1+np.exp(-z))
        return sig
    def initialize(self,X):
        weights = np.zeros((np.shape(X)[1]+1,1))
        X = np.c_[np.ones((np.shape(X)[0],1)),X]
        return weights,X
    def fit(self,X,y,alpha=0.0001,iter=4000):
        weights,X = self.initialize(X)
        def cost(theta):
            z = np.dot(X,theta)
            cost0 = y.T.dot(np.log(self.sigmoid(z)))
            cost1 = (1-y).T.dot(np.log(1-self.sigmoid(z)))
            cost = -((cost1 + cost0))/len(y)
            return cost
        cost_list = np.zeros(iter,)
        for i in range(iter):
            weights = weights - alpha*np.dot(X.T,self.sigmoid(np.dot(X,weights))-np.reshape(y,(len(y),1)))
            cost_list[i] = cost(weights)
        self.weights = weights
        return cost_list
    def predict(self,X):
        z = np.dot(self.initialize(X)[1],self.weights)
        lis = []
        for i in self.sigmoid(z):
            if i>0.5:
                lis.append(1)
            else:
                lis.append(0)
        return lis

class Logisticregression:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        # X=self._normalize(X)

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            # linear_model = np.clip(linear_model, -709.78, 709.78)
            y_predicted = self._sigmoid(linear_model)
            dw = (2/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # X=self._normalize(X)
        linear_model = np.matmul(X, self.weights.T) + self.bias
        # linear_model = np.clip(linear_model, -709.78, 709.78)
        y_predicted = self._softmax(linear_model)
        y_predicted_cls = [1 if i>0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def _softmax(self,x):
        try:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        except:
            print("softmax solution 1 failed")
        try:
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)
        except:
            print("softmax solution 2 failed")

        # Solution 3
        try:
            assert len(x.shape) == 2
            s = np.max(x, axis=1)
            s = s[:, np.newaxis]  # necessary step to do broadcasting
            e_x  = np.exp(x - s)
            div = np.sum(e_x, axis=1)
            div = div[:, np.newaxis]  # dito
            return e_x/div
        except:
            print("softmax solution 3 failed")

    def _normalize(self, x):

        # X --> Input.

        # m-> number of training examples
        # n-> number of features
        m, n = x.shape

        # Normalizing all the n features of X.
        for i in range(n):
            x = (x - x.mean(axis=0)) / x.std(axis=0)

        return x


class LogitRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning

        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent

    def update_weights(self):
        A = 1.0 / (1 + np.exp(- (self.X.dot(self.W) + self.b)))

        # calculate gradients
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function  h( x )

    def predict(self, X):
        Z = 1 / (1 + np.exp(- (X.dot(self.W) + self.b)))
        Y = np.where(Z > 0.5, 1, 0)
        return Y

def plotregression(X_1, y_1, X_2, y_2, X_line, y_line, title):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_1, y_1, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_2, y_2, color=cmap(0.5), s=10)
    plt.plot(X_line, y_line, color='black', linewidth=2, label="Prediction")
    plt.title(title)

def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted
                    )**2)


def acc(y_true, y_predicted):
    return np.sum((y_true==y_predicted)/len(y_true))
def standardize(X_tr):
    for i in range(np.shape(X_tr)[1]):
        X_tr[:,i] = (X_tr[:,i] - np.mean(X_tr[:,i]))/np.std(X_tr[:,i])

from sklearn.datasets import load_digits
digits = load_digits()
# from sklearn.datasets import load_breast_cancer
# digits = load_digits()
type(digits.data)
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(digits.data[0:5],
#                                            digits.target[0:5])):
#     plt.subplot(1, 5, index + 1)
#     plt.imshow(np.reshape(image, (30,30)), cmap=plt.cm.gray)
#     plt.title('Training: %i\n' % label, fontsize = 20)

X,y = digits.data, digits.target
# from sklearn.datasets import make_classification
# X,y = make_classification(n_features=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
# standardize(X_train)
# standardize(X_test)
regressor = Logisticregression(lr=0.0195, n_iters=10000)
model = regressor.fit(X_train, y_train)
logreg_predicted = regressor.predict(X_test)
acc_value = acc(y_test, logreg_predicted)
print("LogReg1 Classification Accuracy:", acc_value)