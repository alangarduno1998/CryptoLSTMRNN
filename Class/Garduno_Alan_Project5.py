import csv
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import xlrd
import numpy as np
from tqdm.notebook import tqdm_notebook
from sklearn.metrics import mean_squared_error, accuracy_score
class NeuralNetwork_3():
    def __init__(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0

    def __sigmoid(self, x):
        return np.tanh(x)
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.

    def forward_propogation(self, x):
        # forward pass - preactivation and activation
        self.x = x
        #activation neuron 1
        self.a1 = self.w1 * self.x + self.b1
        self.h1 = self.__sigmoid(self.a1)

        #activation neuron 2
        self.a2 = self.w2 * self.x + self.b2
        self.h2 = self.__sigmoid(self.a2)

        #activation neuron 3
        self.a3 = self.w3 * self.x +self.b3
        self.h3 = self.__sigmoid(self.a3)
        #output layer neuron 1
        self.a4 = self.w4 * self.h1 + self.w5 * self.h2 + self.w6 * self.h3 + self.b4
        self.h4 = self.__sigmoid(self.a4)
        return self.h3

    def grad(self, x, y):
        # back propagation
        self.forward_propogation(x)
        y=(y+1)/2
        # back propagating the weights and bias between hidden layer and output layer
        self.dw4 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.h1
        self.dw5 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.h2
        self.dw6 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.h3
        self.db4 = (self.h4 - y) * self.h4 * (1 - self.h4)

        # back propagating the weights and bias between input layer and hidden layer
        self.dw1 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w4 * self.h1 * (1 - self.h1) * self.x
        self.dw2 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w5 * self.h2 * (1 - self.h2) * self.x
        self.dw3 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w6 * self.h3 * (1 - self.h3) * self.x

        self.db1 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w4 * self.h1 * (1 - self.h1)
        self.db2 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w5 * self.h2 * (1 - self.h2)
        self.db3 = (self.h4 - y) * self.h4 * (1 - self.h4) * self.w6 * self.h3 * (1 - self.h3)

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=True):

        # initialise w, b
        if initialise:
            self.w1 = np.random.randn()
            self.w2 = np.random.randn()
            self.w3 = np.random.randn()
            self.w4 = np.random.randn()
            self.w5 = np.random.randn()
            self.w6 = np.random.randn()
            self.b1 = 0
            self.b2 = 0
            self.b3 = 0
            self.b4 = 0
        if display_loss:
            loss = {}

        for i in range(epochs):
            dw1, dw2, dw3, dw4, dw5, dw6, db1, db2, db3, db4 = [0] * 10
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3
                db4 += self.db4
            m = X.shape[1]
            self.w1 -= learning_rate * dw1 / m
            self.w2 -= learning_rate * dw2 / m
            self.w3 -= learning_rate * dw3 / m
            self.w4 -= learning_rate * dw4 / m
            self.w5 -= learning_rate * dw5 / m
            self.w6 -= learning_rate * dw6 / m
            self.b1 -= learning_rate * db1 / m
            self.b2 -= learning_rate * db2 / m
            self.b3 -= learning_rate * db3 / m
            self.b4 -= learning_rate * db4 / m
            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = mean_squared_error(Y_pred, Y)

        if display_loss:
            plt.figure(3)
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Part b ii Loss function through training process ')

    def predict(self, X):
        # predicting the results on unseen data
        Y_pred = []
        for x in X:
            y_pred = self.forward_propogation(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)


class NeuralNetwork_20():
    def __init__(self):
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.w4 = np.random.randn()
        self.w5 = np.random.randn()
        self.w6 = np.random.randn()
        self.w7 = np.random.randn()
        self.w8 = np.random.randn()
        self.w9 = np.random.randn()
        self.w10 = np.random.randn()
        self.w11 = np.random.randn()
        self.w12 = np.random.randn()
        self.w12 = np.random.randn()
        self.w13 = np.random.randn()
        self.w14 = np.random.randn()
        self.w15 = np.random.randn()
        self.w16 = np.random.randn()
        self.w17 = np.random.randn()
        self.w18 = np.random.randn()
        self.w19 = np.random.randn()
        self.w20 = np.random.randn()
        self.w21 = np.random.randn()
        self.w22 = np.random.randn()
        self.w23 = np.random.randn()
        self.w24 = np.random.randn()
        self.w25 = np.random.randn()
        self.w26 = np.random.randn()
        self.w27 = np.random.randn()
        self.w28 = np.random.randn()
        self.w29 = np.random.randn()
        self.w30 = np.random.randn()
        self.w31 = np.random.randn()
        self.w32 = np.random.randn()
        self.w33 = np.random.randn()
        self.w34 = np.random.randn()
        self.w35 = np.random.randn()
        self.w36 = np.random.randn()
        self.w37 = np.random.randn()
        self.w38 = np.random.randn()
        self.w39 = np.random.randn()
        self.w40 = np.random.randn()
        self.w41 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0
        self.b7 = 0
        self.b8 = 0
        self.b8 = 0
        self.b9 = 0
        self.b10 = 0
        self.b11 = 0
        self.b12 = 0
        self.b13 = 0
        self.b14 = 0
        self.b15 = 0
        self.b16 = 0
        self.b17 = 0
        self.b18 = 0
        self.b19 = 0
        self.b20 = 0
        self.b21 = 0
    def __sigmoid(self, x):
        return np.tanh(x)
    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.

    def forward_propogation(self, x):
        # forward pass - preactivation and activation
        self.x = x
        #activation neuron 1-20
        self.a1 = self.w1 * self.x + self.b1
        self.h1 = self.__sigmoid(self.a1)
        self.a2 = self.w2 * self.x + self.b2
        self.h2 = self.__sigmoid(self.a2)
        self.a3 = self.w3 * self.x +self.b3
        self.h3 = self.__sigmoid(self.a3)
        self.a4 = self.w4 * self.x + self.b4
        self.h4 = self.__sigmoid(self.a4)
        self.a5 = self.w5 * self.x + self.b5
        self.h5 = self.__sigmoid(self.a5)
        self.a6 = self.w6 * self.x + self.b6
        self.h6 = self.__sigmoid(self.a6)
        self.a7 = self.w7 * self.x +self.b7
        self.h7 = self.__sigmoid(self.a7)
        self.a8 = self.w8 * self.x + self.b8
        self.h8 = self.__sigmoid(self.a8)
        self.a9 = self.w9 * self.x + self.b9
        self.h9 = self.__sigmoid(self.a9)
        self.a10 = self.w10 * self.x + self.b10
        self.h10 = self.__sigmoid(self.a10)
        self.a11 = self.w11 * self.x +self.b11
        self.h11 = self.__sigmoid(self.a11)
        self.a12 = self.w12 * self.x + self.b12
        self.h12 = self.__sigmoid(self.a12)
        self.a13 = self.w13 * self.x + self.b13
        self.h13 = self.__sigmoid(self.a13)
        self.a14 = self.w14 * self.x + self.b14
        self.h14 = self.__sigmoid(self.a14)
        self.a15 = self.w15 * self.x +self.b15
        self.h15 = self.__sigmoid(self.a15)
        self.a16 = self.w16 * self.x + self.b16
        self.h16 = self.__sigmoid(self.a16)
        self.a17 = self.w17 * self.x + self.b17
        self.h17 = self.__sigmoid(self.a17)
        self.a18 = self.w18 * self.x + self.b18
        self.h18 = self.__sigmoid(self.a18)
        self.a19 = self.w19 * self.x +self.b19
        self.h19 = self.__sigmoid(self.a19)
        self.a20 = self.w20 * self.x + self.b20
        self.h20 = self.__sigmoid(self.a20)

        #output layer neuron 1
        self.a21 = self.w21 * self.h1 + self.w22 * self.h2 + self.w23 * self.h3 + self.w24 * self.h4 + \
                   self.w25 * self.h5 + self.w26 * self.h6 + self.w27 * self.h7 + self.w28 * self.h8 + \
                   self.w29 * self.h9 + self.w30 * self.h10 + self.w31 * self.h11 + self.w32 * self.h12 + \
                   self.w33 * self.h13 + self.w34 * self.h14 + self.w35 * self.h15 + self.w36 * self.h16 + \
                   self.w37 * self.h17 + self.w38 * self.h18 + self.w39 * self.h19 + self.w40 * self.h20 + \
                   self.b21
        self.h21 = self.__sigmoid(self.a21)
        return self.h21

    def grad(self, x, y):
        # back propagation
        self.forward_propogation(x)
        y=(y+1)/2
        # back propagating the weights and bias between hidden layer and output layer
        self.dw21 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h1
        self.dw22 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h2
        self.dw23 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h3
        self.dw24 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h4
        self.dw25 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h5
        self.dw26 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h6
        self.dw27 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h7
        self.dw28 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h8
        self.dw29 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h9
        self.dw30 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h10
        self.dw31 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h11
        self.dw32 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h12
        self.dw33 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h13
        self.dw34 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h14
        self.dw35 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h15
        self.dw36 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h16
        self.dw37 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h17
        self.dw38 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h18
        self.dw39 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h19
        self.dw40 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h20

        self.db21 = (self.h21 - y) * self.h21 * (1 - self.h21)

        # back propagating the weights and bias between input layer and hidden layer
        self.dw1 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h1 * self.w21 * (1 - self.h1) * self.x
        self.dw2 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h2 * self.w22 * (1 - self.h2) * self.x
        self.dw3 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h3 * self.w23 * (1 - self.h3) * self.x
        self.dw4 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h4 * self.w24 * (1 - self.h4) * self.x
        self.dw5 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h5 * self.w25 * (1 - self.h5) * self.x
        self.dw6 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h6 * self.w26 * (1 - self.h6) * self.x
        self.dw7 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h7 * self.w27 * (1 - self.h7) * self.x
        self.dw8 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h8 * self.w28 * (1 - self.h8) * self.x
        self.dw9 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h9 * self.w29 * (1 - self.h9) * self.x
        self.dw10 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h10 * self.w30 * (1 - self.h10) * self.x
        self.dw11 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h11 * self.w31 * (1 - self.h11) * self.x
        self.dw12 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h12 * self.w32 * (1 - self.h12) * self.x
        self.dw13 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h13 * self.w33 * (1 - self.h13) * self.x
        self.dw14 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h14 * self.w34 * (1 - self.h14) * self.x
        self.dw15 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h15 * self.w35 * (1 - self.h15) * self.x
        self.dw16 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h16 * self.w36 * (1 - self.h16) * self.x
        self.dw17 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h17 * self.w37 * (1 - self.h17) * self.x
        self.dw18 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h18 * self.w38 * (1 - self.h18) * self.x
        self.dw19 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h19 * self.w39 * (1 - self.h19) * self.x
        self.dw20 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h20 * self.w40 * (1 - self.h20) * self.x

        self.db1 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h1 * self.w21 * (1 - self.h1)
        self.db2 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h2 * self.w22 * (1 - self.h2)
        self.db3 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h3 * self.w23 * (1 - self.h3)
        self.db4 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h4 * self.w24 * (1 - self.h4)
        self.db5 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h5 * self.w25 * (1 - self.h5)
        self.db6 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h6 * self.w26 * (1 - self.h6)
        self.db7 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h7 * self.w27 * (1 - self.h7)
        self.db8 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h8 * self.w28 * (1 - self.h8)
        self.db9 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h9 * self.w29 * (1 - self.h9)
        self.db10 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h10 * self.w30 * (1 - self.h10)
        self.db11 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h11 * self.w31 * (1 - self.h11)
        self.db12 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h12 * self.w32 * (1 - self.h12)
        self.db13 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h13 * self.w33 * (1 - self.h13)
        self.db14 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h14 * self.w34 * (1 - self.h14)
        self.db15 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h15 * self.w35 * (1 - self.h15)
        self.db16 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h16 * self.w36 * (1 - self.h16)
        self.db17 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h17 * self.w37 * (1 - self.h17)
        self.db18 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h18 * self.w38 * (1 - self.h18)
        self.db19 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h19 * self.w39 * (1 - self.h19)
        self.db20 = (self.h21 - y) * self.h21 * (1 - self.h21) * self.h20 * self.w40 * (1 - self.h20)

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=True):

        # initialise w, b
        if initialise:
            self.w1 = np.random.randn()
            self.w2 = np.random.randn()
            self.w3 = np.random.randn()
            self.w4 = np.random.randn()
            self.w5 = np.random.randn()
            self.w6 = np.random.randn()
            self.w7 = np.random.randn()
            self.w8 = np.random.randn()
            self.w9 = np.random.randn()
            self.w10 = np.random.randn()
            self.w11 = np.random.randn()
            self.w12 = np.random.randn()
            self.w12 = np.random.randn()
            self.w13 = np.random.randn()
            self.w14 = np.random.randn()
            self.w15 = np.random.randn()
            self.w16 = np.random.randn()
            self.w17 = np.random.randn()
            self.w18 = np.random.randn()
            self.w19 = np.random.randn()
            self.w20 = np.random.randn()
            self.w21 = np.random.randn()
            self.w22 = np.random.randn()
            self.w23 = np.random.randn()
            self.w24 = np.random.randn()
            self.w25 = np.random.randn()
            self.w26 = np.random.randn()
            self.w27 = np.random.randn()
            self.w28 = np.random.randn()
            self.w29 = np.random.randn()
            self.w30 = np.random.randn()
            self.w31 = np.random.randn()
            self.w32 = np.random.randn()
            self.w33 = np.random.randn()
            self.w34 = np.random.randn()
            self.w35 = np.random.randn()
            self.w36 = np.random.randn()
            self.w37 = np.random.randn()
            self.w38 = np.random.randn()
            self.w39 = np.random.randn()
            self.w40 = np.random.randn()
            self.w41 = np.random.randn()
            self.b1 = 0
            self.b2 = 0
            self.b3 = 0
            self.b4 = 0
            self.b5 = 0
            self.b6 = 0
            self.b7 = 0
            self.b8 = 0
            self.b8 = 0
            self.b9 = 0
            self.b10 = 0
            self.b11 = 0
            self.b12 = 0
            self.b13 = 0
            self.b14 = 0
            self.b15 = 0
            self.b16 = 0
            self.b17 = 0
            self.b18 = 0
            self.b19 = 0
            self.b20 = 0
            self.b21 = 0
        if display_loss:
            loss = {}

        for i in range(epochs):
            dw1, dw2, dw3, dw4, dw5, dw6 , dw7, dw8, dw9, dw10, dw11, dw12, dw13, dw14, dw15, dw16 , dw17, dw18, dw19, \
            dw20, dw21, db1, db2, db3, db4, db5, db6, db7, db8, db9, db10, db11, db12, db13, db14, db15, db16, db17,\
            db18, db19, db20, db21 = [0] * 42
            for x, y in zip(X, Y):
                self.grad(x, y)
                dw1 += self.dw1
                dw2 += self.dw2
                dw3 += self.dw3
                dw4 += self.dw4
                dw5 += self.dw5
                dw6 += self.dw6
                dw7 += self.dw7
                dw8 += self.dw8
                dw9 += self.dw9
                dw10 += self.dw10
                dw11 += self.dw11
                dw12 += self.dw12
                dw13 += self.dw13
                dw14 += self.dw14
                dw15 += self.dw15
                dw16 += self.dw16
                dw17 += self.dw17
                dw18 += self.dw18
                dw19 += self.dw19
                dw20 += self.dw20
                dw21 += self.dw21
                db1 += self.db1
                db2 += self.db2
                db3 += self.db3
                db4 += self.db4
                db5 += self.db5
                db6 += self.db6
                db7 += self.db7
                db8 += self.db8
                db9 += self.db9
                db10 += self.db10
                db11 += self.db11
                db12 += self.db12
                db13 += self.db13
                db14 += self.db14
                db15 += self.db15
                db16 += self.db16
                db17 += self.db17
                db18 += self.db18
                db19 += self.db19
                db20 += self.db20
                db21 += self.db21
            m = X.shape[1]
            self.w1 -= learning_rate * dw1 / m
            self.w2 -= learning_rate * dw2 / m
            self.w3 -= learning_rate * dw3 / m
            self.w4 -= learning_rate * dw4 / m
            self.w5 -= learning_rate * dw5 / m
            self.w6 -= learning_rate * dw6 / m
            self.w7 -= learning_rate * dw7 / m
            self.w8 -= learning_rate * dw8 / m
            self.w9 -= learning_rate * dw9 / m
            self.w10 -= learning_rate * dw10 / m
            self.w11 -= learning_rate * dw11 / m
            self.w12 -= learning_rate * dw12 / m
            self.w13 -= learning_rate * dw13 / m
            self.w14 -= learning_rate * dw14 / m
            self.w15 -= learning_rate * dw15 / m
            self.w16 -= learning_rate * dw16 / m
            self.w17 -= learning_rate * dw17 / m
            self.w18 -= learning_rate * dw18 / m
            self.w19 -= learning_rate * dw19 / m
            self.w20 -= learning_rate * dw20 / m
            self.w21 -= learning_rate * dw21 / m
            self.b1 -= learning_rate * db1 / m
            self.b2 -= learning_rate * db2 / m
            self.b3 -= learning_rate * db3 / m
            self.b4 -= learning_rate * db4 / m
            self.b5 -= learning_rate * db5 / m
            self.b6 -= learning_rate * db6 / m
            self.b7 -= learning_rate * db7 / m
            self.b8 -= learning_rate * db8 / m
            self.b9 -= learning_rate * db9 / m
            self.b10 -= learning_rate * db10 / m
            self.b11 -= learning_rate * db11 / m
            self.b12 -= learning_rate * db12 / m
            self.b13 -= learning_rate * db13 / m
            self.b14 -= learning_rate * db14 / m
            self.b15 -= learning_rate * db15 / m
            self.b16 -= learning_rate * db16 / m
            self.b17 -= learning_rate * db17 / m
            self.b18 -= learning_rate * db18 / m
            self.b19 -= learning_rate * db19 / m
            self.b20 -= learning_rate * db20 / m
            self.b21 -= learning_rate * db21 / m
            if display_loss:
                Y_pred = self.predict(X)
                loss[i] = mean_squared_error(Y_pred, Y)

        if display_loss:
            plt.figure(5)
            plt.plot(loss.values())
            plt.xlabel('Epochs')
            plt.ylabel('Mean Squared Error')
            plt.title('Part b iii Loss function through training process ')

    def predict(self, X):
        # predicting the results on unseen data
        Y_pred = []
        for x in X:
            y_pred = self.forward_propogation(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


# Initialization of the neural network parameters
# Initialized all the weights in the range of between 0 and 1
# Bias values are initialized to 0
def initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures):
    W1 = np.random.randn(neuronsInHiddenLayers, inputFeatures)
    W2 = np.random.randn(outputFeatures, neuronsInHiddenLayers)
    b1 = np.zeros((neuronsInHiddenLayers, 1))
    b2 = np.zeros((outputFeatures, 1))

    parameters = {"W1": W1, "b1": b1,
                  "W2": W2, "b2": b2}
    return parameters


# Forward Propagation
def forwardPropagation(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    # A1 = sigmoid(Z1)
    A1 = tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    # A2 = sigmoid(Z2)
    A2 = tanh(Z2)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2)
    # logprobs = np.multiply(np.log((A2+np.min(A2)+1)), Y) + np.multiply(np.log(1 - (A2+np.min(A2)+1)), (1 - Y))
    logprobs = np.multiply(np.log((A2+1)/2), (Y+1)) + np.multiply(np.log(1 - A2), (1 - (Y+1)/2))
    cost = -np.sum(logprobs) / m
    return cost, cache, A2


# Backward Propagation
def backwardPropagation(X, Y, cache):
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2) = cache

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    # dZ1 = np.multiply(dA1, A1 * (1 - A1))
    dZ1 = np.multiply(dA1, (1 - A1**2))

    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    gradients = {"dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients


# Updating the weights based on the negative gradients
def updateParameters(parameters, gradients, learningRate):
    parameters["W1"] = parameters["W1"] - learningRate * gradients["dW1"]
    parameters["W2"] = parameters["W2"] - learningRate * gradients["dW2"]
    parameters["b1"] = parameters["b1"] - learningRate * gradients["db1"]
    parameters["b2"] = parameters["b2"] - learningRate * gradients["db2"]
    return parameters

if __name__ == "__main__":

    #Seed the random number generator
    random.seed(20)
    X = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1]])  # XOR input
    Y = np.array([[-1, 1, 1, -1]])  # XOR output
    # Define model parameters
    neuronsInHiddenLayers = 2  # number of hidden layer neurons (2)
    inputFeatures = X.shape[0]  # number of input features (2)
    outputFeatures = Y.shape[0]  # number of output features (1)
    parameters = initializeParameters(inputFeatures, neuronsInHiddenLayers, outputFeatures)
    epoch = 100000
    learningRate = 0.01
    losses = np.zeros((epoch, 1))

    for i in range(epoch):
        losses[i, 0], cache, A2 = forwardPropagation(X, Y, parameters)
        gradients = backwardPropagation(X, Y, cache)
        parameters = updateParameters(parameters, gradients, learningRate)

    # Evaluating the performance
    plt.figure(1)
    plt.plot(losses)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.title("Loss function through training process part a")

    # Testing
    # X = np.array([[1, 1, 0, 0], [0, 1, 0, 1]])  # XOR input
    X = np.array([[1, 1, -1, -1], [-1, 1, -1, 1]])  # XOR input
    cost, _cache, A2 = forwardPropagation(X, Y, parameters)
    # prediction = (A2 > 0.5) * 1.0
    prediction = []
    for i in A2[0]:
        if (i >= 0.0):
            prediction.append(1)
        else:
            # prediction = 0
            prediction.append(-1)

    # print(A2[0])
    # print(prediction)
    # print("For input1", [i for i in X[0]], "and input2", [i for i in X[1]], "output is",
    #       [i for i in prediction])  # ['{:.2f}'.format(i) for i in x])
    plt.figure(2)
    boundary1 = np.linspace(-1.2, 1.2, 100)
    plt.plot(boundary1, boundary1 + 1.5, c='black')
    plt.plot(boundary1, boundary1 - 1.5, c='black')
    plt.fill_between(x=boundary1, y1=boundary1 + 1.5, y2=boundary1 + 3, alpha=.2, color='blue')
    plt.fill_between(x=boundary1, y1=boundary1 - 1.5, y2=boundary1 - 3, alpha=.2, color='blue')
    plt.fill_between(x=boundary1, y1=boundary1 + 1.5, y2=boundary1 - 1.5, alpha=.2, color='red')
    for i in range(len(prediction)):
        if (prediction[i] == -1):
            # print(X[0][i])
            # print(X[1][i])
            plt.scatter(X[0][i], X[1][i], c='red', s=100)
        else:
            plt.scatter(X[0][i], X[1][i], c='blue', s=100)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.1, 1.1)
    plt.title("decision surface part a")
    # Create layer 1 (4 neurons, 3 for hidden layer and 1 for output layer, each with 3 inputs)


    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    CSV_NAME = "Proj5Dataset.csv"
    filename = "Proj5Dataset.xlsx"
    df = pandas.read_excel(filename, engine='openpyxl')  # use engine, xlrd==2.0.1 cant open xlsx file
    df.to_csv(CSV_NAME, index=None, header=True)
    dataset = numpy.genfromtxt(CSV_NAME, delimiter=",", dtype=numpy.float, missing_values="", filling_values="0")
    dataset = dataset[dataset[:, dataset.shape[1] - 1] != 0]
    dataset = dataset[dataset[:, dataset.shape[1] - 1] != 0]  # remove them stupid 0's
    n_samples, n_features = dataset.shape[0], dataset.shape[1] - 1
    training_set_inputs, training_set_outputs = dataset[:, 0:n_features], dataset[:, n_features]

    NN=NeuralNetwork_3()
    NN.fit(training_set_inputs, training_set_outputs, epochs=2000, learning_rate=0.01, display_loss=True)
    pred_training = NN.predict(training_set_inputs)
    pred_binarised_training = (pred_training >= 0).astype("int").ravel()
    pred_val = NN.predict(training_set_inputs)
    pred_binarised_val = (pred_val >= 0).astype("int").ravel()
    accuracy_train = mean_squared_error(pred_binarised_training, training_set_outputs)
    accuracy_val = mean_squared_error(pred_binarised_val, pred_val)
    # model performance
    print("Training accuracy for 3 neurons", round(accuracy_train, 2))
    print("Validation accuracy for 3 neurons", round(accuracy_val, 2))
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    plt.figure(4)
    plt.scatter(training_set_inputs[:, 0], training_set_inputs[:, 0], c=pred_binarised_training, cmap=my_cmap,
                s=15 * (np.abs(pred_binarised_training - training_set_outputs) + .2))
    plt.title("part b ii model with input data")

    NN=NeuralNetwork_20()
    NN.fit(training_set_inputs, training_set_outputs, epochs=2000, learning_rate=0.01, display_loss=True)
    pred_training = NN.predict(training_set_inputs)
    pred_binarised_training = (pred_training >= 0).astype("int").ravel()
    pred_val = NN.predict(training_set_inputs)
    pred_binarised_val = (pred_val >= 0).astype("int").ravel()
    accuracy_train = mean_squared_error(pred_binarised_training, training_set_outputs)
    accuracy_val = mean_squared_error(pred_binarised_val, pred_val)
    # model performance
    print(" Part bTraining accuracy for 20 neurons", round(accuracy_train, 2))
    print("Part b Validation accuracy for 20 neurons", round(accuracy_val, 2))
    my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])
    plt.figure(6)
    plt.scatter(training_set_inputs[:, 0], training_set_inputs[:, 0], c=pred_binarised_training, cmap=my_cmap,
                s=15 * (np.abs(pred_binarised_training - training_set_outputs) + .2))
    plt.title("part b iii model with input data")
    plt.show()