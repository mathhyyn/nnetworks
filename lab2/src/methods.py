import numpy as np
import matplotlib.pyplot as plt

def gradient(perc):
    x = perc.x_train
    y = perc.y_train
    num = perc.n_train

    epohs = []
    errors = []

    for step in range(5):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(num):  # обучение на n тестах
            perc.layers[0].out = x[i]
            out = perc.forward(x[i])

            # последний слой
            l = perc.layers[-1]
            lastDelta = [perc.dloss(y[i][j], l.out[j]) for j in range(l.n_neurons)] # * l.derivative(l.XW[j]))
            lastDelta = np.array(lastDelta)
            gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
            perc.layers[-1].w += l.lr * gradient

            # скрытые слои
            # for l in perc.layers[1:-1:-1]:
            for l_i in range(len(perc.layers) - 2, 0, -1):
                l = perc.layers[l_i]
                # средневзвешенная delta выходов
                sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                lastDelta = np.array(delta)
                gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                perc.layers[l_i].w += l.lr * gradient

    plt.plot(epohs, errors)
    plt.show()



def FletcherReeves(perc):
    x = perc.x_train
    y = perc.y_train
    num = perc.n_train

    epohs = []
    errors = []

    for step in range(5):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(num):  # обучение на n тестах
            perc.layers[0].out = x[i]
            out = perc.forward(x[i])

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]

                if (l_i == len(perc.layers) - 1):
                # последний слой
                    lastDelta = [perc.dloss(y[i][j], l.out[j]) for j in range(l.n_neurons)] # * l.derivative(l.XW[j]))
                    lastDelta = np.array(lastDelta)
                    gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
                    #perc.layers[-1].w += l.lr * gradient
                else:
                # скрытые слои
                    # средневзвешенная delta выходов
                    sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                    delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                    lastDelta = np.array(delta)
                    gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                    #perc.layers[l_i].w += l.lr * gradient

                if len(l.prev_grad) == 0 or len(l.prev_d) == 0:
                    l.prev_grad = [row[:] for row in gradient]
                    l.prev_d = l.lr * gradient

                W = [[min(gradient[i][j] ** 2 / (l.prev_grad[i][j] ** 2 + 1e-6), 1) for j in range(l.n_input)] for i in range (l.n_neurons)]
                d = l.prev_grad - np.array(W) * l.prev_d
                l.w += l.lr * d
                l.prev_grad = [row[:] for row in gradient]
                l.prev_d = np.array([row[:] for row in d])

    plt.plot(epohs, errors)
    plt.show()


def bfgs(perc):
    x = perc.x_train
    Y = perc.y_train
    num = perc.n_train

    epohs = []
    errors = []

    for step in range(5):
        if step % 1 == 0:
            print(step)
            epohs.append(step)
            errors.append(perc.countErr())

        for i in range(num):  # обучение на n тестах
            perc.layers[0].out = x[i]
            out = perc.forward(x[i])

            for l_i in range(len(perc.layers) - 1, 0, -1):
                l = perc.layers[l_i]

                if (l_i == len(perc.layers) - 1):
                # последний слой
                    lastDelta = [perc.dloss(Y[i][j], l.out[j]) for j in range(l.n_neurons)] # * l.derivative(l.XW[j]))
                    lastDelta = np.array(lastDelta)
                    gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
                    #perc.layers[-1].w += l.lr * gradient
                else:
                # скрытые слои
                    # средневзвешенная delta выходов
                    sum = np.dot(perc.layers[l_i + 1].w.T, lastDelta)
                    delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                    lastDelta = np.array(delta)
                    gradient = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                    #perc.layers[l_i].w += l.lr * gradient  

                if len(l.prev_grad) == 0:
                    l.prev_grad = np.array([row[:] for row in gradient])
                    perc.layers[l_i].w += l.lr * gradient
                else:
                    p = - np.dot(l.H, gradient.reshape(-1, 1))
                    I = np.eye(l.n_neurons * l.n_input)
                    s = l.lr * p
                    l.w += s.reshape(l.n_neurons, l.n_input)
                    y = l.prev_grad - gradient
                    #s = s.reshape(-1, 1)
                    y = y.reshape(-1, 1)
                    rho = 1 / np.dot(y.T, s)
                    A = I - rho * np.dot(s, y.T)
                    B = I - rho * np.dot(y, s.T)
                    l.H = np.dot(np.dot(A, l.H), B) + rho * np.dot(s, s.T)
                    l.prev_grad = [row[:] for row in gradient]

    plt.plot(epohs, errors)
    plt.show()