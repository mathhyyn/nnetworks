import os
import numpy as np
from src.methods import gradient, SGD, NAG, Adagrad, Adam, plt

import gzip
import struct

import random

n_pixels = 784  # 28*28

def relu(x):
    return x if x > 0 else 0
def drelu(x):
    return 1 if x > 0 else 0

def softmax(xs):
    maxX = max(xs)
    exp_values = np.exp(xs - maxX)

    sum_exp_values = np.sum(exp_values)
    return [ex / sum_exp_values for ex in exp_values]

# среднквадратичная ф.п
def mse(y0, y):
    return 1 / 2 * (y0 - y) ** 2
def dmse(y0, y):
    return y - y0
    res = []
    for i in range(len(y)):
        res.append(sum(y[i] - y0[i]) / len(y[i]))
    return res


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
        self.prev_grad = []
        self.vt = np.zeros_like(self.w) # NAG
        self.LR = np.zeros_like(self.w) # Adagrad
        self.m = np.zeros_like(self.w) # Adam
        self.v = np.zeros_like(self.w) # Adam

    def forward(self, x):
        self.XW = np.dot(self.w, x)
        if self.activation == softmax:
            self.out = self.activation(self.XW)
        else:
            self.out = np.array([self.activation(xw) for xw in self.XW])
        return self.out


class Perceptron:
    def __init__(self, x_train, y_train, lr = 0.01):
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

    def forward(self, x):
        self.layers[0].out = x.copy()
        out = x.copy()
        i = 1
        for l in self.layers[1:]:
            out = l.forward(out)
            i += 1
        self.out = out
        return out
    
    def change_w_nag(self, k):
        for l in self.layers[1:]:
            l.w += k * l.vt * 0.9

    def gradient(self):
        gradient(self)
    def SGD(self):
        SGD(self)
    def NAG(self, log=True):
        NAG(self, log)
    def Adagrad(self):
        Adagrad(self)
    def Adam(self):
        Adam(self)

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
    
    '''def countGradient(self, y):
        gradient = []
        for l_i in range(len(self.layers) - 1, 0, -1):
            l = self.layers[l_i]

            if (l_i == len(self.layers) - 1):
                lastDelta = l.out - y
            else:
                l = self.layers[l_i]
                sum = np.dot(self.layers[l_i + 1].w.T, lastDelta)
                lastDelta = np.array([sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)])
                
            gradient.insert(0, np.outer(lastDelta, self.layers[l_i - 1].out))
        return gradient'''

    def find_answ(res):
        num = 0
        min = 1
        for i in range(len(res)):
            if abs(1 - res[i]) < min:
                min = abs(1 - res[i])
                num = i
        return num

    # проверка ответов
    def checkCorrectness(self, x = [], y = [], log = True):
        if len(x) == 0:
            x = self.x_train
            y = self.y_train

        num = len(x)
        correct_num = 0
        for i in range(num):
            res = [0] * 10
            mas = [0] * 10

            res = self.forward(x[i])

            for j in range(10):
                mas[j] = round(res[j], 2)

            predicted = np.argmax(res)
            expected = np.argmax(y[i])
            if expected == predicted:
                correct_num += 1

            if i > num - 5 and log:
                print(mas)
                print(y[i])
                print(expected, "--->", predicted, "\n")

        res = correct_num / num * 100
        if log:
            print(res, "%% correctness")
        return res


def create_Y_ans(Y_first):
    Y_res = []
    for y in Y_first:
        mas = np.zeros(10)
        mas[y] = 1
        Y_res.append(mas)
    return  np.array(Y_res)


data_folder = os.path.join(os.getcwd(), "data")

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack("I", gz.read(4))
        n_items = struct.unpack(">I", gz.read(4))
        if not label:
            n_rows = struct.unpack(">I", gz.read(4))[0]
            n_cols = struct.unpack(">I", gz.read(4))[0]
            res = np.frombuffer(
                gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(os.path.join(
    data_folder, "train-images.gz"), False) / 255.0
Y_train = load_data(os.path.join(
    data_folder, "train-labels.gz"), True).reshape(-1)
X_test = load_data(os.path.join(data_folder, "test-images.gz"), False) / 255.0
Y_test = load_data(os.path.join(
    data_folder, "test-labels.gz"), True).reshape(-1)

train_len = len(X_train)
n_tests = 500
X_first = X_train[:n_tests]
Y_first = Y_train[:n_tests]

Y_res = create_Y_ans(Y_first)

def createPerc(size=1, neuron_num=[10]):
    perc = Perceptron(X_first, Y_res)
    perc.neuron_num = neuron_num
    prev = n_pixels
    for i in range(size):
        perc.add_layer(n_input=prev, n_neurons=neuron_num[i], lr=0.01)
        prev = neuron_num[i]
    perc.add_layer(n_input=prev, n_neurons=10, func_act=softmax, lr=0.01)
    return perc

if __name__ == "__main__":
    perc1 = Perceptron(X_first, Y_res)
    perc1.add_layer(n_input=n_pixels, n_neurons=10, lr=0.01)
    perc1.add_layer(n_neurons=10, func_act=softmax, lr=0.01)



    W1 = [row.copy() for row in perc1.layers[1].w]
    W2 = [row.copy() for row in perc1.layers[2].w]
    perc1.gradient()
    perc1.checkCorrectness(X_first, Y_res)
    perc1.layers[1].w = [row[:] for row in W1]
    perc1.layers[2].w = [row[:] for row in W2]
    perc1.SGD()
    perc1.checkCorrectness(X_first, Y_res)
    perc1.layers[1].w = [row[:] for row in W1]
    perc1.layers[2].w = [row[:] for row in W2]
    perc1.NAG()
    perc1.checkCorrectness(X_first, Y_res)
    perc1.layers[1].w = [row[:] for row in W1]
    perc1.layers[2].w = [row[:] for row in W2]
    perc1.Adagrad()
    perc1.checkCorrectness(X_first, Y_res)
    perc1.layers[1].w = [row[:] for row in W1]
    perc1.layers[2].w = [row[:] for row in W2]
    perc1.Adam()
    perc1.checkCorrectness(X_first, Y_res)
    plt.legend()
    plt.show()

    '''print(perc1.layers[-2].w)
    print(perc1.layers[-1].w)'''

    #Y_res2 = create_Y_ans(Y_test)
    #perc1.checkCorrectness(X_test[:500], Y_res2[:500])
