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
from src.methods import gradient1

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

def loss(y0, y):
  return 1 / 2 * (y0 - y)**2

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
        #self.w = np.array([[1/n_input for _ in range(n_input)] for _ in range(n_neurons)])
        self.w = np.array([[random.uniform(1/(4*n_input), 4/n_input) for _ in range(n_input)] for _ in range(n_neurons)])
        #self.w = np.array([[random.uniform(0.001, 0.01) for _ in range(n_input)] for _ in range(n_neurons)])
        #print(self.w)
        self.activation = activation
        self.derivative = derivative
        #self.vect_act = np.vectorize(activation)
        self.XW = []
        self.out = []
        self.lr = lr
    def forward(self, x):
        '''if self.activation == softmax:
            out = []
            for n_i in range(len(self.n_neurons)):
                xw = x * self.w[n_i]
                sft = self.activation(xw)
                out.append(sft)
            self.out = out[:]
        else:'''
        self.XW = np.dot(self.w, x)
        self.out = [self.activation(xw) for xw in self.XW]
        return self.out[:]

class Perceptron:
    def __init__(self, x_train, y_train, lossfunc):
        self.layers = []
        self.loss = lossfunc
        self.out = []
        self.x_train = x_train
        self.y_train = y_train
        self.lastDelta = []
    
    def add_layer(self, n_neurons, n_input = -1, func_act = relu, dfunc_act = drelu, lr = 0.1):
        if len(self.layers) == 0:
            l0 = Layer(n_input, 0, func_act, dfunc_act, lr) # создание 0го - входного слоя
            self.layers.append(l0) 
        n_input = n_input if n_input > 0 else len(self.layers[-1].w) # кол-во нейронов предыдущего слоя
        self.layers.append(Layer(n_neurons, n_input, func_act, dfunc_act, lr))

    def forward(self, x):
        out = x[:]
        #self.layers[0].out = out[:]
        i = 1
        for l in self.layers[1:]:
            out = l.forward(out[:])
            #print(i, out)
            i+=1
        self.out = out[:]
        return out
    
    def gradient(self):
        gradient1(self)
    




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

Y_res = []
for y in Y_first:
    mas = np.zeros(10)
    mas[y] = 1
    Y_res.append(mas)


perc1 = Perceptron(X_first, Y_res, loss)
perc1.add_layer(n_neurons=10, n_input=15, func_act=relu, dfunc_act=drelu, lr = 0.01)
perc1.add_layer(n_neurons=10, func_act=relu, dfunc_act=drelu, lr = 0.01)
#perc1.add_layer(n_neurons=10, func_act=relu, dfunc_act=drelu, lr = 0.01)
#perc1.add_layer(n_neurons=10, lr = 0.01)
#perc1.add_layer(n_neurons=10)
#perc1.add_layer(n_neurons=10)
'''perc1.add_layer(n_neurons=10, n_input=784, func_act=sigm, dfunc_act=dsigm)
perc1.add_layer(n_neurons=10, func_act=sigm, dfunc_act=dsigm)'''
#perc1.add_layer(n_neurons=10)
perc1.gradient()

print(perc1.layers[1].w)
print(perc1.layers[2].w)

