import numpy as np #pip3 install numpy
import matplotlib.pyplot as plt #pip3 install matplotlib
import pandas as pd  # pip3 install pandas, openpyxl, xlrd

class ClosedForm:
    def fit(self, X, t):
        #self.coef_ = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(t)
        #self.coef_ = (np.linalg.inv(X.T @ X) @ X.T @ t)
        self.coef_ = np.linalg.pinv(X).dot(t)
        return self

    def predict(self, X):
        return X.dot(self.coef_)


class GradientDescent:
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
    def erms(self):
        erms = np.sqrt(self.weights/self.n_iters)
        return erms

def plotregression(X_1, y_1, X_2, y_2, X_line, y_line, label, color):
    cmap = plt.get_cmap('inferno')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_1, y_1, color='red', s=10)
    # m2 = plt.scatter(X_2, y_2, color=cmap(0.9), s=10)
    line1,=plt.plot(X_line, y_line, color = "{}".format(color), linewidth=2, label="{}".format(label))
    plt.legend(handles=[line1], loc="upper right")
    plt.xlabel("Weight"), plt.ylabel("Horsepower")
    plt.title("Python's carbig dataset")


def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted
                    )**2)


def acc(y_true, y_predicted):
    return np.sum((y_true==y_predicted)/len(y_true))
CSV_NAME = "Test.csv"
df = pd.read_excel("proj1Dataset.xlsx", engine='openpyxl') # use engine, xlrd==2.0.1 cant open xlsx file
df.to_csv (CSV_NAME, index = None, header=True)
data = np.genfromtxt(CSV_NAME, delimiter=",", dtype=np.float, missing_values="", filling_values="0")
data = data[data[:,data.shape[1]-1] != 0] # remove them stupid 0's
n_samples, n_features = data.shape[0], data.shape[1] -1
X,y = data[:,0:n_features], data[:,n_features]


cf_regr = ClosedForm()
cf_regr.fit(X, y)
cf_predicted = cf_regr.predict(X)
cf_mse = mse(y, cf_predicted)
print("Closed Form MSE Error:", cf_mse)
plotregression(X, y, X, y, X, cf_predicted, label="Closed Form", color = "green")


gd_regr = GradientDescent(lr=0.000000195, n_iters=1000)#0.000000195, 386.05487538376707
gd_regr.fit(X, y)
gd_predicted = gd_regr.predict(X)
gd_mse = mse(y, gd_predicted)
print("Gradient MSE Error:", gd_mse)
gd_rmse = gd_regr.erms()
print("Gradient RMSE Error:", gd_rmse)
plotregression(X, y, X, y, X, gd_predicted, label="GradientDescent", color = "blue")
# if (gd_mse> cf_mse):
#     print('keep tuning')
# else:
#     print('its aight ')


plt.show()