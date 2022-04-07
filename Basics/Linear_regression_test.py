import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


def plotregression(X_1, y_1, X_2, y_2, X_line, y_line):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_1, y_1, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_2, y_2, color=cmap(0.5), s=10)
    plt.plot(X_line, y_line, color='black', linewidth=2, label="Prediction")

# def plotperceptron(p,X_1, y_1, X_2, y_2, X_line, y_line, label):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(1,1,1)
#     plt.scatter(X_1[:, 0], X_1[:, 1], marker='o', c=y_1)
#
#     x0_1 = np.amin(X_1[:,0])
#     x0_2 = np.amax(X_1[:,0])
#
#     x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
#     x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
#
#     ax.plot([x0_1, x0_2], [x1_1, x1_2], label="{} Prediction".format(label))
#
#     ymin = np.amin(X_1[:,1])
#     ymax = np.amax(X_1[:,1])
#     ax.set_ylim([ymin-3, ymax+3])
#
#
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted
                    )**2)


def acc(y_true, y_predicted):
    return np.sum((y_true==y_predicted)/len(y_true))


# X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
#
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color= "b", marker="o", s= 30)

from Regression import LinearRegression

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
regressor = LinearRegression(lr=0.001)
regressor.fit(X_train, y_train)
linreg_predicted = regressor.predict(X_test)
mse_value = mse(y_test, linreg_predicted)
print("LinReg MSE Error:", mse_value)
y_pred_line = regressor.predict(X)
plotregression(X_train, y_train, X_test, y_test, X, y_pred_line)
plt.show()
# from Regression import LogisticRegression
# bc = datasets.load_breast_cancer()
# X,y = bc.data, bc.target
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
# regressor = LogisticRegression(lr=0.001)
# regressor.fit(X_train, y_train)
# logreg_predicted = regressor.predict(X_test)
# acc_value = acc(y_test, logreg_predicted)
# print("LogReg Classification Accuracy:", acc_value)
# y_pred_line = regressor.predict(X)
# print("X_train shape:",X_train.shape,"\n y_train shape:", y_train.shape)
# plotregression(X_train, y_train, X_test, y_test, X, y_pred_line, "Logistic")
# plt.show()

# from Regression import Perceptron
# bc = datasets.load_breast_cancer()
# X,y = datasets.make_blobs(n_samples=150,n_features=2, centers=2, cluster_std=1.05, random_state=2)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
# p = Perceptron(lr=0.001)
# p.fit(X_train, y_train)
# p_predicted = p.predict(X_test)
# acc_value = acc(y_test, p_predicted)
# print("Perceptron Classification Accuracy:", acc_value)
# p_pred_line = p.predict(X)
# print("X_train shape:",X_train.shape,"\n y_train shape:", y_train.shape)
# plotperceptron(p,X_train, y_train, X_test, y_test, X, p_pred_line, "Perceptron")
# plt.show()

# def visualize_svm(label):
#     def get_hyperplane_value(x,w,b, offset):
#         return (-w[0] * x * b + offset) / w[1]
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     plt.scatter(X[:,0], X[:,1], marker='o', c=y)
#
#     x0_1 = np.amin(X[:, 0])
#     x0_2 = np.amax(X[:, 0])
#
#     x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
#     x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)
#
#     x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
#     x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)
#
#     x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
#     x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)
#
#     ax.plot([x0_1, x0_2], [x1_1, x1_2], label="{} y--".format(label))
#     ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], label="{} k".format(label))
#     ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], label="{} k".format(label))
#
#
#
#     x_min = np.amin(X[:, 1])
#     x_max = np.amax(X[:, 1])
#     ax.set_ylim([x_min - 3, x_max + 3])
#
#     plt.show()
# def SVM_run():
#     from Regression import SVM
#     X,y = datasets.make_blobs(n_samples=50,n_features=2, centers=2, cluster_std=1.05, random_state=40)
#     y = np.where(y == 0, -1, 1)
#     clf = SVM()
#     clf.fit(X, y)
#     print(clf.w, clf.b)
#     visualize_svm("SVM")
#SVM_run()

# def PCA_run():
#     from Regression import PCA
#     data = datasets.load_iris()
#     X, y = data.data, data.target
#
#     # 150, 4
#
#     pca = PCA(2) # porject onto two primary principal components
#     pca.fit(X)
#     X_projected = pca.transform(X)
#
#     print('Sahpe of X:', X.shape)
#     print('Shape of transformed X:', X_projected.shape)
#     x1 = X_projected[:,0]
#     x2 = X_projected[:,1]
#
#     plt.scatter(x1, x2, edgecolors='none', c=y, alpha=0.8,
#                 cmap=plt.cm.get_cmap('viridis',3))
#     plt.xlabel('Principal component 1')
#     plt.ylabel('Principal component 2')
#     plt.colorbar
#     plt.show()
#PCA_run()

# from Regression import DecisionTree
# data = datasets.load_breast_cancer()
# X,y = data.data, data.target
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
# dtree = DecisionTree(max_depth=10)
# dtree.fit(X_train, y_train)
# y_predicted = dtree.predict(X_test)
# acc_value = acc(y_test, y_predicted)
# print("Decision Tree Classification Accuracy:", acc_value)

# from Regression import RandomForest
# data = datasets.load_breast_cancer()
# X,y = data.data, data.target
# print(X.shape, y.shape)
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
# rndmf = RandomForest(n_trees=3)
# rndmf.fit(X_train, y_train)
# y_predicted = rndmf.predict(X_test)
# acc_value = acc(y_test, y_predicted)
# print("Decision Tree Classification Accuracy:", acc_value)