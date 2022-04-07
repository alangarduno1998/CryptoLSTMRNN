import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy.special import softmax


from scipy.io import loadmat
mnist = loadmat("../Class/mnist/mnist-original.mat")
mnist_data = mnist["data"].T
mnist_label = mnist["label"][0]
# mnist = fetch_openml("mnist_784", version=1)
X,y = mnist_data, mnist_label

# W is (M+1)xK -> 2x2
W = np.random.randn(3,2)

# X is Nx(M+1) -> Nx2
df = pd.read_csv(r"D:\Documents\Downloads\classification.csv")
x = df.iloc[:,:-1]
x = np.hstack((x, np.ones((x.shape[0],1))))
# fix t to make it one-hot encoded
classes = df.loc[:,'success'].astype('int64')

t = np.array([9, 9])
for test in classes:
    if test:
        t = np.vstack((t, [1, 0]))
    else:
        t = np.vstack((t, [0, 1]))
t = np.delete(t, 0, axis=0)

phi = 0.0001
n = x.shape[0] # num samples
k = W.shape[1] # num classes
for _ in range(10000):
    a = np.dot(x, W)
    y = softmax(a, axis=1)
    for j in range(k):
        sum_error_gradient = 0
        for i in range(n):
            sum_error_gradient = sum_error_gradient + np.dot((y[i][j] - t[i][j]), x[i])
        error = sum_error_gradient.reshape(3,1)
        W[:, j:j+1] = W[:, j:j+1] - (phi * error)
print(W)

y = softmax(np.dot(x, W), axis=1)

Y = np.array([9,9])
for preds in y:
    for i in range(len(preds)):
        if preds[i] > 0.5000:
            arr = np.zeros(len(preds))
            arr[i] = 1
            Y = np.vstack((Y, arr))
Y = np.delete(Y, 0, axis=0)

print(Y)

# caluclate confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns

cf_matrix= multilabel_confusion_matrix(t,Y, labels=[True, False])

print(cf_matrix)
# Recall= TP/(TP+FN)
# Precision=TP/(TP+FP)
# Accuracy=(TP+TN)/Total
clf_report=classification_report(Y,t)

print(clf_report)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


fig, ax = plt.subplots(2, 2, figsize=(12, 7))
labels = ["".join("c" + str(i)) for i in range(0, 4)]

for axes, cfs_matrix, label in zip(ax.flatten(), cf_matrix, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

fig.tight_layout()
plt.show()
