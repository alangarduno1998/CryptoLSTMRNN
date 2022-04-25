import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sb
import csv
from scipy.io import loadmat, savemat
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle, time

## take input dataset and model from user
flag=0
user_input_dataset = input("Enter the path to your dataset: ")
    # will check the file extension, if .tif then mnist and if .png then c.elegans

for image in os.listdir(user_input_dataset):
    if image.endswith('.tif'):
        flag=1
    elif user_input_dataset.endswith('.png'):
        flag=2
        print('here')


if (flag==2):
    W = pd.read_csv("worm_weights.csv")
    W = np.array(W)
    print('here')
    basepath_1 = user_input_dataset
    images = []
    filenames = []

    for image in os.listdir(basepath_1):
        print('here')
        filenames.append(image)
        if os.path.isfile(os.path.join(basepath_1, image)):
            img = mpimg.imread(os.path.join(basepath_1, image))
            # remove line below to insert 101x101 images
            arr_img = img.reshape([(img.shape[0] * img.shape[1])])
            images.append(arr_img)

    x = np.array(images).reshape(len(images), img.shape[0] * img.shape[1])
    x = np.hstack((x, np.ones((len(x), 1))))
    print(x.shape)
    y_test = softmax(np.dot(x, W), axis=1)

    y_pred = np.array([9, 9])
    for preds in y_test:
        index = np.argmax(preds)
        arr = np.zeros(2)
        arr[index] = 1
        y_pred = np.vstack((y_pred, arr))
    y_pred = np.delete(y_pred, 0, axis=0)
    print("Predictions from the C. Elegans model.\n")
    for i, files in enumerate(filenames):
        worm = np.array([1, 0])
        label = "no worm"
        if np.array_equal(y_pred[i], worm):
            label = "worm"

        print("Filename: {}, Label: {}".format(filenames[i], label))


if (flag==1):
    X_test = []
    for image in os.listdir(user_input_dataset):
        if os.path.isfile(os.path.join(user_input_dataset, image)):
            if os.path.join(user_input_dataset, image).endswith('.txt'):
                Y_test=np.loadtxt(fname=os.path.join(user_input_dataset, image))
                Y_test=Y_test.reshape(Y_test.shape[0], 1).astype(int)
            else:
                img = mpimg.imread(os.path.join(user_input_dataset, image))
                # remove line below to insert 101x101 images
                arr_img = img.reshape([(img.shape[0] * img.shape[1])])
                X_test.append(arr_img)

    t0=time.time()
    X_test = np.array(X_test)
    def normalize(data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data_normalized = (data - mean) / std
        return data_normalized
    X_test = normalize(X_test)
    filename_input = 'mnist_trained_model.sav'
    models_list = pickle.load(open(filename_input, 'rb'))

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
    print(pred_label)
    print(Y_test[:,-1])
    print(accuracy_score(pred_label, Y_test[:,-1]))
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
    with open('mnist_predicted_labels.csv','w') as csvfile:
        writer=csv.writer(csvfile, delimiter=';')
        writer.writerows(zip(Y_test[:,-1],pred_label))


    ## Output - Total number of images for each label
    n0_classes_list = []
    print('Total number of images for each label')
    for i in range(10):
        print('Label {}: {} images '.format(i, sum((pred_label==i).astype(int))))
    runtime= t1+time.time()-t2
    print('Total training time is:{} minutes and {}seconds'.format(int(runtime/60), round(runtime%60,3)))
