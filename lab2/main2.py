import os
import numpy as np
import matplotlib.pyplot as plt
from src.gradient import gradient1

import gzip
import struct

from dataclasses import dataclass
from pprint import pprint

n_pixels = 784 # 28*28

def relu(x):
  return x if x > 0 else 0

def lost(y0, y):
  return 1 / 2 * (y0 - y)**2

class Layer:
    def __init__(self, n_neurons, n_input, activation):
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.w = np.array([[0.01 for _ in range(n_input)] for _ in range(n_neurons)])
        self.activation = activation
        self.vect_act = np.vectorize(activation)
        self.XW = []
        self.out = []
        self.lr = 0.01
    def __str__(self) -> str:
        return f"{self.n}\n{self.w}\n{self.activation}"
    def forward(self, x):
        self.XW = np.dot(self.w, x)
        self.out = self.vect_act(self.XW)
        return self.out

class Perceptron:
    def __init__(self, x_train, y_train, lostfunc):
        self.layers = []
        self.lost = lostfunc
        self.out = []
        self.x_train = x_train
        self.y_train = y_train
        self.lastDelta = []
    
    def add_layer(self, n_neurons, n_input = -1, lostfunc = relu):
        if len(self.layers) == 0:
            l0 = Layer(n_input, 0, lostfunc) # создание 0го - входного слоя
            self.layers.append(l0) 
        n_input = n_input if n_input > 0 else len(self.layers[-1].w) # кол-во нейронов предыдущего слоя
        self.layers.append(Layer(n_neurons, n_input, lostfunc))

    def forward(self, x):
        out = x
        for l in self.layers[1:]:
            out = l.forward(out)
        self.out = out
        return out
    
    def gradient(self):
        gradient1(self)
    


data_folder = os.path.join(os.getcwd(), 'data')

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            print(n_cols*n_rows)
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
Y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)

train_len = len(X_train)
X_first = X_train[:10]
Y_first = Y_train[:10]
print(Y_first)

Y_res = []
for y in Y_first:
    mas = np.zeros(10)
    mas[y] = 1
    Y_res.append(mas)


perc1 = Perceptron(X_first, Y_res, lost)
perc1.add_layer(n_neurons=10, n_input=784)
perc1.add_layer(n_neurons=10)

perc1.gradient()