import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import csv
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle, time

t0=time.time()
## input dataset and model
mnist = loadmat("../Class/mnist/mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]
# print(mnist_data)

filename_input = 'mnist_trained_model.sav'
models_list = pickle.load(open(filename_input, 'rb'))

image_size_px = int(np.sqrt(mnist_data.shape[1]))
# print("The images size is (", image_size_px, 'x', image_size_px, ')')

def normalize(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data_normalized = (data - mean)/std
    return data_normalized

mnist_data_normalized = normalize(mnist_data)

X_train, X_test, Y_train, Y_test = train_test_split(mnist_data_normalized, mnist_label, test_size=0.20, random_state=42)
Y_train = Y_train.reshape(Y_train.shape[0],1)
Y_test = Y_test.reshape(Y_test.shape[0],1)
Y_train_list, Y_test_list = [],[]
for i in range(10):
    Y_train_list.append((Y_train==i).astype(int))
    Y_test_list.append((Y_test==i).astype(int))

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def predict(X,W,B):
    Yhat_prob = sigmoid(np.dot(X,W)+B)
    Yhat = np.round(Yhat_prob).astype(int)
    return Yhat, Yhat_prob


def one_vs_all(data, models_list):
    pred_matrix = np.zeros((data.shape[0],10))
    for i in range(len(models_list)):
        W = models_list[i]['weights']
        B = models_list[i]['bias']
        Yhat, Yhat_prob = predict(data,W,B)
        pred_matrix[:,i] = Yhat_prob.T
    max_prob_vec = np.amax(pred_matrix, axis=1, keepdims=True)
    pred_matrix_max_prob = (pred_matrix == max_prob_vec).astype(int)
    labels=[]
    for j in range(pred_matrix_max_prob.shape[0]):
        idx = np.where(pred_matrix_max_prob[j,:]==1)
        labels.append(idx)
    labels = np.vstack(labels).flatten()
    return labels

pred_label = one_vs_all(X_test, models_list)

## Testing information presented as confusion matrix
conf_matrix = confusion_matrix(Y_test, pred_label)

def plot_cm(mat,y_ture,ax,case):
    if case == 0:
        df_cm = pd.DataFrame(mat, columns=np.unique(y_ture), index = np.unique(y_ture))
        df_cm.index.name = 'True Label'
        df_cm.columns.name = 'Predicted Label'
        sb.heatmap(df_cm, cmap="Blues", cbar=False, annot=True,annot_kws={"size": 10}, ax=ax)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)
    else:
        l_lab=['Goalkeeper','Defender','Midfielder','Forward']
        df_cm = pd.DataFrame(mat, columns=np.array(l_lab), index = np.unique(l_lab))
        df_cm.index.name = 'True Label'
        df_cm.columns.name = 'Predicted Label'
        sb.heatmap(df_cm, cmap="Blues", cbar=False, annot=True,annot_kws={"size": 10}, ax=ax)
        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)

plt.figure(figsize=(10,7))
ax1 = plt.subplot(111)
plt.title('Confusion matrix of the one-vs-all logistic regression classifier', fontsize=16)
plot_cm(conf_matrix, Y_test, ax1,0)
t1=time.time()-t0
plt.show()

t2=time.time()
## Output - Two column list
with open('mnist_imagenames_labels.csv','w') as csvfile:
    writer=csv.writer(csvfile, delimiter=';')
    writer.writerows(zip(Y_test[:,-1],pred_label))


## Output - Total number of images for each label
n0_classes_list = []
print('Total number of images for each label')
for i in range(10):
    print('Label {}: {} images '.format(i, sum((pred_label==i).astype(int))))
runtime= t1+time.time()-t2
print('Total training time is:{} minutes and {}seconds'.format(int(runtime/60), round(runtime%60,3)))

