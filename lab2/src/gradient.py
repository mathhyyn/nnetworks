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
            grads = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
            perc.layers[-1].w += l.lr * grads

            # скрытые слои
            # for l in perc.layers[1:-1:-1]:
            for l_i in range(len(perc.layers) - 2, 0, -1):
                l = perc.layers[l_i]
                # средневзвешенная delta выходов
                sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)
                delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                lastDelta = np.array(delta)
                grads = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                perc.layers[l_i].w += l.lr * grads

    plt.plot(epohs, errors)
    plt.show()



def FletcherReeves(perc):
    x = perc.x_train
    y = perc.y_train
    num = perc.n_train

    '''def LOSS():
        s = 0
        for i in range(num):  # обучение на n тестах
            out = perc.forward(x[i])
            s += sum([perc.dloss(y[i][j], out[j]) for j in range(perc.layers[-1].n_neurons)])
        return s/num

    def func_for_min(a):
        perc.aFR = a'''


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
                    grads = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
                    #perc.layers[-1].w += l.lr * grads
                else:
                # скрытые слои
                    # средневзвешенная delta выходов
                    sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)
                    delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                    lastDelta = np.array(delta)
                    grads = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                    #perc.layers[l_i].w += l.lr * grads

                if len(l.pred_grads) == 0 or len(l.pred_w) == 0:
                    l.pred_grads = [row[:] for row in grads]
                    l.pred_w = l.lr * grads

                beta = [[min(grads[i][j] ** 2 / (l.pred_grads[i][j] ** 2 + 1e-6), 1) for j in range(l.n_input)] for i in range (l.n_neurons)]
                delta_param = l.pred_grads - np.array(beta) * l.pred_w
                l.w += l.lr * delta_param
                l.pred_grads = [row[:] for row in grads]
                l.pred_w = [row[:] for row in delta_param]

    plt.plot(epohs, errors)
    plt.show()



def bfgs(perc):
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
                    grads = -np.dot(np.transpose([lastDelta]), [perc.layers[-2].out])
                    #perc.layers[-1].w += l.lr * grads
                else:
                # скрытые слои
                    # средневзвешенная delta выходов
                    sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)
                    delta = [sum[j] * l.derivative(l.XW[j]) for j in range(l.n_neurons)]
                    lastDelta = np.array(delta)
                    grads = -np.dot(np.transpose([lastDelta]), [perc.layers[l_i - 1].out])
                    #perc.layers[l_i].w += l.lr * grads  

                I = np.eye(l.n_neurons, l.n_input)
                pk = - np.dot(l.H, l.pred_grads)
                l.w += l.lr*pk
                sk = l.w - l.pred_w
                yk = grads - l.pred_grads
                k = 1/(np.transpose(yk)*sk)
                l.H = (I - k*sk*np.transpose(yk))*l.H(I - k*yk*np.transpose(sk)) + k*sk*np.transpose(sk)
                l.pred_grad = [row[:] for row in grads]
                l.pred_w = [row[:] for row in l.w]

    plt.plot(epohs, errors)
    plt.show()
