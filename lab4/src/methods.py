import numpy as np
import matplotlib.pyplot as plt
import random

def gradient(perc):
    x = perc.x_train
    y = perc.y_train

    epohs = []
    errors = []

    for step in range(15):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(perc.n_train):  # обучение на n тестах
            out = perc.forward(x[i])

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]

                if (l_i == len(perc.layers) - 1):
                # последний слой
                    lastDelta = l.out - y[i]
                    #lastDelta = np.array([perc.dloss(y[i][j], l.out[j]) for j in range(l.n_neurons)])
                else:
                # скрытые слои
                    l = perc.layers[l_i]
                    # средневзвешенная delta выходов
                    sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                    lastDelta = np.array([sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)])
                    
                #gradient = np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                gradient = np.outer(lastDelta, perc.layers[l_i - 1].out)
                perc.layers[l_i].w -= l.lr * gradient
                

    plt.plot(epohs, errors, label="gradient")
    #plt.show()

'''
def gradient(perc):
    x = perc.x_train
    y = perc.y_train
    num = perc.n_train

    epohs = []
    errors = []

    for step in range(10):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        out = []
        for i in range(num):
            perc.layers[0].out = x[i]
            out.append(perc.forward(x[i]))
        out = np.array(out)

        for l_i in range(len(perc.layers) - 1, 0, -1):
            l = perc.layers[l_i]

            if (l_i == len(perc.layers) - 1):
            # последний слой
                lastDelta = perc.dloss(y.T, out.T)
                #print(lastDelta)
                #lastDelta = np.array([perc.dloss(y[i][j], l.out[j]) for j in range(l.n_neurons)])
            else:
            # скрытые слои
                l = perc.layers[l_i]
                # средневзвешенная delta выходов
                sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                lastDelta = np.array([sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)])
                
            #gradient = np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
            gradient = np.outer(lastDelta, perc.layers[l_i - 1].out)
            perc.layers[l_i].w -= l.lr * gradient
                

    plt.plot(epohs, errors, label=f"{perc.loss.__name__}")
    #plt.show()
'''
    

def SGD(perc):
    x1 = perc.x_train
    y1 = perc.y_train

    epohs = []
    errors = []
    
    batch_size = 200

    for step in range(15):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        #rand_batch = np.random.randint(0, num - batch_size)
        batch = random.sample(list(zip(x1, y1)), batch_size)
        for x, y in batch:
            out = perc.forward(x)

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]

                if (l_i == len(perc.layers) - 1):
                    lastDelta = l.out - y
                else:
                    l = perc.layers[l_i]
                    sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                    lastDelta = np.array([sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)])
                    
                gradient = np.outer(lastDelta, perc.layers[l_i - 1].out)
                perc.layers[l_i].w -= l.lr * gradient
                

    plt.plot(epohs, errors, label="SGD")


def NAG(perc):
    x = perc.x_train
    y = perc.y_train

    epohs = []
    errors = []

    for step in range(15):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(perc.n_train):
            perc.change_w_nag(-1)
            out = perc.forward(x[i])
            perc.change_w_nag(1)

            gradient = perc.countGradient(y[i])

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]
                
                l.vt += 0.00001 * gradient[l_i - 1]
                l.w -= l.vt
                

    plt.plot(epohs, errors, label="NAG")


def Adagrad(perc):
    x = perc.x_train
    y = perc.y_train

    epohs = []
    errors = []

    for step in range(15):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(perc.n_train):
            out = perc.forward(x[i])

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]

                if (l_i == len(perc.layers) - 1):
                    lastDelta = l.out - y[i]
                else:
                    l = perc.layers[l_i]
                    sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                    lastDelta = np.array([sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)])
                    
                gradient = np.outer(lastDelta, perc.layers[l_i - 1].out)
                perc.layers[l_i].w -= l.lr * gradient
        norm = np.linalg.norm(gradient)
        l.lr /= 1 if norm == 0 else norm
                

    plt.plot(epohs, errors, label="Adagrad")


def Adam(perc):
    pass
