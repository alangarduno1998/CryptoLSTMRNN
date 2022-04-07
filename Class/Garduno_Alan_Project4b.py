import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def acc(y_true, y_predicted):
    return np.sum((y_true==y_predicted)/len(y_true))


from sklearn.datasets import load_digits
digits = load_digits()


X,y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression(max_iter=10000)

regressor.fit(X_train, y_train)
logreg_predicted = regressor.predict(X_test)
acc_value = acc(y_test, logreg_predicted)
print("LogReg2 Classification Accuracy:", acc_value)


from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1)
from scipy.io import loadmat
mnist = loadmat("../Class/mnist/mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]
# mnist = fetch_openml("mnist_784", version=1)
X,y = mnist_data, mnist_label
train_img, test_img, train_lbl, test_lbl = train_test_split(X,y, test_size=0.2, random_state=1234)
# plt.figure(figsize=(20,4))
# for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
#  plt.subplot(1, 5, index + 1)
#  plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
#  plt.title('Training: %i\n' % label, fontsize = 20)
from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression(solver = 'lbfgs', max_iter=1000)

regressor.fit(train_img, train_lbl)
logreg_predicted = regressor.predict(test_img)
acc_value = acc(test_lbl, logreg_predicted)
print("LogReg3 Classification Accuracy:", acc_value)