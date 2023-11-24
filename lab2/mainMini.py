from PIL import Image, ImageDraw #Подключим необходимые библиотеки.
import sys

def getImage(N):
  image = Image.open(f"tests/{N}.png") #Открываем изображение.
  width = image.size[0] #Определяем ширину.
  height = image.size[1] #Определяем высоту.
  pix = image.load() #Выгружаем значения пикселей.

  if width != 3 and height != 5:
    sys.exit()

  x = [0] * 15

  for i in range(width):
    for j in range(height):
      x[j * width + i] = 0 if pix[i, j] == (0,0,0,0) else 1

  return x

import os
import numpy as np
import matplotlib.pyplot as plt
from src.methods import gradient, bfgs

import gzip
import struct

from dataclasses import dataclass
from pprint import pprint
import random

n_pixels = 784 # 28*28

def lin(x):
    return x
def dlin(x):
    return 1
def relu(x):
  return x if x > 0 else 0
def drelu(x):
  return 1 if x > 0 else 0
def sigm(x):
  return 1 / (1 + np.exp(-x))
def dsigm(x):
  return np.exp(-x) / (1 + np.exp(-x)) ** 2

def mse(y0, y):
    return 1 / 2 * (y0 - y) ** 2
def dmse(y0, y):
    return y - y0

def softmax(x):
    exp_values = np.exp(x)
    sum_exp_values = np.sum(exp_values)
    return exp_values / sum_exp_values

def dsoftmax(x):
    exp_values = np.exp(x)
    sum_exp_values = np.sum(exp_values)
    softmax_values = exp_values / sum_exp_values
    softmax_derivative_values = softmax_values * (1 - softmax_values)
    return softmax_derivative_values

class Layer:
    def __init__(self, n_neurons, n_input, activation, derivative, lr):
        self.n_neurons = n_neurons
        self.n_input = n_input
        # self.w = np.array([[1/n_input for _ in range(n_input)] for _ in range(n_neurons)])
        # self.w = np.array([[0.01 for _ in range(n_input)] for _ in range(n_neurons)])
        self.w = np.array([[random.uniform(1 / (2 * n_input), 2 / n_input)
                          for _ in range(n_input)] for _ in range(n_neurons)])
        self.activation = activation
        self.derivative = derivative
        self.XW = []
        self.out = []
        self.lr = lr
        self.prev_grad = [] #FR
        self.prev_d = [] # FR
        self.H = np.eye(n_neurons * n_input) #BFGS

    def forward(self, x):
        self.XW = np.dot(self.w, x)
        if self.activation == softmax:
            self.out = self.activation(self.XW)
        else:
            self.out = [self.activation(xw) for xw in self.XW]
        return self.out


class Perceptron:
    def __init__(self, x_train, y_train):
        self.layers = []
        self.loss = mse
        self.dloss = dmse
        self.out = []
        self.x_train = x_train
        self.y_train = y_train
        self.n_train = len(x_train)
        self.lastDelta = []

    def add_layer(self, n_neurons, n_input=-1, func_act=relu, dfunc_act=drelu, lr=0.1):
        if len(self.layers) == 0:
            # создание 0го - входного слоя
            l0 = Layer(n_input, 0, func_act, dfunc_act, lr)
            self.layers.append(l0)
        n_input = (
            n_input if n_input > 0 else len(self.layers[-1].w)
        )  # кол-во нейронов предыдущего слоя
        self.layers.append(Layer(n_neurons, n_input, func_act, dfunc_act, lr))

    def set_loss(self, lossfunc_name):
        if lossfunc_name == "cross_entr":
            self.loss = cross_entr
        elif lossfunc_name == "KL":
            self.loss = kl_divergence

    def forward(self, x):
        out = x[:]
        # self.layers[0].out = out[:]
        i = 1
        for l in self.layers[1:]:
            out = l.forward(out)
            # print(i, out)
            i += 1
        self.out = out
        return out

    def gradient(self):
        gradient(self)
    def BFGS(self):
        bfgs(self)

    def find_answ(res):
        num = 0
        min = 1
        for i in range(len(res)):
            if abs(1 - res[i]) < min:
                min = abs(1 - res[i])
                num = i
        return num

    def countErr(self):
        all = 0
        for i in range(self.n_train):  # обучение на n тестах
            e = 0
            res = self.forward(self.x_train[i])
            for j in range(10):  # 10 нейронов
                e += self.loss(self.y_train[i][j], res[j])
            e /= 10  # среднее
            all += e
        all /= self.n_train
        return all

    def checkCorrectness(self, x, y):
        # проверка ответов
        num = len(x)
        correct_num = 0
        for i in range(num):
            res = [0] * 10
            mas = [0] * 10

            res = self.forward(x[i])

            for j in range(10):
                mas[j] = round(res[j], 2)

            predicted = np.argmax(res)
            print(mas)
            print(y[i])
            expected = np.argmax(y[i])

            print(expected, "--->", predicted, "\n")
            if expected == predicted:
                correct_num += 1

        print(correct_num / num * 100, "%% correctness")


def create_Y_ans(Y_first):
    Y_res = []
    for y in Y_first:
        mas = np.zeros(10)
        mas[y] = 1
        Y_res.append(mas)
    return Y_res
    

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = []
Y_train = []
for i in range(10):
  Y_train.append(i)
  X_train.append(getImage(i))

train_len = len(X_train)
'''n_tests = 30
X_first = X_train[:n_tests]
Y_first = Y_train[:n_tests]'''
X_first = X_train[:]
Y_first = Y_train[:]


Y_res = create_Y_ans(Y_first)
perc1 = Perceptron(X_first, Y_res)
perc1.add_layer(n_neurons=10, n_input=15, func_act=relu, dfunc_act=drelu, lr = 0.01)
#perc1.add_layer(n_neurons=10, func_act=relu, dfunc_act=drelu, lr = 0.01)
#perc1.add_layer(n_neurons=10, func_act=relu, dfunc_act=drelu, lr = 0.01)
#perc1.add_layer(n_neurons=10, lr = 0.01)
#perc1.add_layer(n_neurons=10)
#perc1.add_layer(n_neurons=10)
'''perc1.add_layer(n_neurons=10, n_input=784, func_act=sigm, dfunc_act=dsigm)
perc1.add_layer(n_neurons=10, func_act=sigm, dfunc_act=dsigm)'''
#perc1.add_layer(n_neurons=10)
perc1.BFGS()

print(perc1.layers[1].w)
#print(perc1.layers[2].w)

perc1.checkCorrectness(X_first, Y_res)

