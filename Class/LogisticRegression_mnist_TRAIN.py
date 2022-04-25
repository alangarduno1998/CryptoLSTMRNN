import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import csv
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import time

t0=time.time()
# input dataset
mnist = loadmat("mnist/mnist-original.mat") # dataset with a bunch of different numbers
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0] # classes of numbers from 0-9
# print(mnist_data)

image_size_px = int(np.sqrt(mnist_data.shape[1])) # input image sizes pix by pix
# print("The images size is (", image_size_px, 'x', image_size_px, ')')

def mnist_random_example():
    idx = np.random.randint(70000)
    exp = mnist_data[idx].reshape(image_size_px,image_size_px)
    # print("The number in the image below is:", mnist_label[idx])
    # plt.imshow(exp) # verify visual data information

mnist_random_example()

def normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_normalized = (data - mean)/std
    return data_normalized

mnist_data_normalized = normalize(mnist_data)

X_train, X_test, Y_train, Y_test = train_test_split(mnist_data_normalized, mnist_label, test_size=0.7, random_state=42) # training data and test data split, rnadomized
Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)
print("The shape of the training set feature matrix is:", X_train.shape)
print("The shape of the training label vector is:", Y_train.shape)
print("The shape of the test set feature matrix is:", X_test.shape)
print("The shape of the test label vector is:", Y_test.shape)
Y_train_list, Y_test_list = [],[]
for i in range(10):
    Y_train_list.append((Y_train==i).astype(int)) # y_train data for classes y_train = [digit0, digit1,... digit 9]
    Y_test_list.append((Y_test==i).astype(int))

def initializer(nbr_features):
    W = np.zeros((nbr_features,1))
    B = 0
    return W, B

def sigmoid(x):
    s = 1/(1+np.exp(-x)) # separates values into sections based on sigmoid, 0 or 1
    return s


def ForwardBackProp(X, Y, W, B):
    m = X.shape[0]
    dw = np.zeros((W.shape[0], 1))
    dB = 0
    # calculating the cost/loss function
    Z = np.dot(X, W) + B
    Yhat = sigmoid(Z)
    J = -(1 / m) * (np.dot(Y.T, np.log(Yhat)) + np.dot((1 - Y).T, np.log(1 - Yhat))) # losssssss function

    #partial derivates of cost function
    dW = (1 / m) * np.dot(X.T, (Yhat - Y))
    dB = (1 / m) * np.sum(Yhat - Y)
    return J, dW, dB

def predict(X,W,B):
    Yhat_prob = sigmoid(np.dot(X,W)+B)
    Yhat = np.round(Yhat_prob).astype(int)
    return Yhat, Yhat_prob


def gradient_descent(X, Y, W, B, alpha, max_iter):
    i = 0
    RMSE = 1
    cost_history = []

    # setup toolbar
    toolbar_width = 20
    sys.stdout.write("[%s]" % ("" * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

    while (i < max_iter) & (RMSE > 10e-6):
        J, dW, dB = ForwardBackProp(X, Y, W, B)
        W = W - alpha * dW
        B = B - alpha * dB
        cost_history.append(J)
        Yhat, _ = predict(X, W, B)
        RMSE = np.sqrt(np.mean(Yhat - Y) ** 2)
        i += 1
        if i % 50 == 0:
            sys.stdout.write("=")
            sys.stdout.flush()

    sys.stdout.write("]\n")  # this ends the progress bar
    return cost_history, W, B, i


def LogRegModel(X_train, X_test, Y_train, Y_test, alpha, max_iter):
    nbr_features = X_train.shape[1]
    W, B = initializer(nbr_features)
    cost_history, W, B, i = gradient_descent(X_train, Y_train, W, B, alpha, max_iter)
    Yhat_train, _ = predict(X_train, W, B)
    Yhat, _ = predict(X_test, W, B)

    train_accuracy = accuracy_score(Y_train, Yhat_train)
    test_accuracy = accuracy_score(Y_test, Yhat)
    conf_matrix = confusion_matrix(Y_test, Yhat, normalize='true')

    model = {"weights": W,
             "bias": B,
             "train_accuracy": train_accuracy,
             "test_accuracy": test_accuracy,
             "confusion_matrix": conf_matrix,
             "cost_history": cost_history}
    return model


print('Progress bar: 1 step each 50 iteration')
model_0 = LogRegModel(X_train, X_test, Y_train_list[0], Y_test_list[0], alpha=0.01, max_iter=1000)
print('Training completed!')

print('The training accuracy of the model',model_0['train_accuracy'])
print('The test accuracy of the model',model_0['test_accuracy'])

def check_random_pred(datum,Y,model,label):
    W = model['weights']
    B = model['bias']
    Yhat, _ = predict(datum,W,B)
    if Yhat == 1:
        pred_label = label
    else:
        pred_label = 'Not '+ label
    if Y == 1:
        true_label = label
    else:
        true_label = 'Not '+ label
    print("The number in the image below is:", true_label, ' and predicted as:', pred_label)
    image = datum.reshape(image_size_px,image_size_px)
    plt.imshow(image)

idx = np.random.randint(X_test.shape[0])
datum = X_test[idx]
Y = Y_test_list[0][idx]
check_random_pred(datum,Y,model_0, '0')

models_list=[]
models_name_list=['model_0','model_1','model_2','model_3','model_4','model_5','model_6',
                 'model_7','model_8','model_9']

print('Training of a classifier for each digit:')
for i in range(10):
    print('Training of the model: ', models_name_list[i],', to recognize the digit: ',i)
    print('Training progress bar: 1 step each 50 iteration')
    model = LogRegModel(X_train, X_test, Y_train_list[i], Y_test_list[i], alpha=0.01, max_iter=1000)
    print('Training completed!')
    print('Accuracy:', model['test_accuracy'])
    print('-'*60)
    models_list.append(model)

accuracy_list=[]
for i in range(len(models_list)):
    accuracy_list.append(models_list[i]['test_accuracy'])
ove_vs_all_accuracy=np.mean(accuracy_list)
print('The accuracy of the Onve-Vs-All model is:', ove_vs_all_accuracy)

#save mnist model to file
filename_output = 'mnist_trained_model.sav'
pickle.dump(models_list, open(filename_output, 'wb'))
runtime=time.time()-t0
print('Total training time is:{} minutes and {}seconds'.format(int(runtime/60), round(runtime%60,3)))

