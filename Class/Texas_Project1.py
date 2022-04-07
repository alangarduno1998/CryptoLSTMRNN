import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd


def closedW(data, target):
    # Closed form: W* = (X(T)X)^(-1)X(T)t
    # Horsepower = t

    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    print(data)

    targ = target.to_numpy()
    targ = targ.reshape(data.shape[0], 1)

    dot1 = np.dot(data.T, data)
    dot2 = np.dot(data.T, targ)

    closedW = np.dot(np.linalg.pinv(dot1), dot2)

    return closedW


# Will have to change this to the file location of your dataset.
dataset = pd.read_excel('proj1Dataset.xlsx', engine='openpyxl')
print(dataset.head())

dataset = dataset.dropna()

weight = dataset[['Weight']]
horsepower = dataset[['Horsepower']]
# weight with 1s appended, x,1
Data = np.append(weight, np.ones((dataset.shape[0], 1)), axis=1)
Data = Data[:,0:Data.shape[1]-1]
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.plot('Weight', 'Horsepower', 'ro', data=dataset)

closed = np.dot(np.linalg.pinv(Data).dot(horsepower))
y = np.dot(Data, closed)
plt.plot(dataset, closed, 'g-')

plt.show()