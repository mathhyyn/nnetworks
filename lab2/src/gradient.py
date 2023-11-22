import numpy as np
import matplotlib.pyplot as plt

def gradient1(perc):
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
            lastDelta = []

            for j in range(l.n_neurons):
                lastDelta.append(perc.dloss(y[i][j], l.out[j])) # * l.derivative(l.XW[j]))
                for k in range(l.n_input):
                    o_k = perc.layers[-2].out[k]
                    perc.layers[-1].w[j][k] -= l.lr * o_k * lastDelta[j]
            lastDelta = np.array(lastDelta)

            # скрытые слои
            # for l in perc.layers[1:-1:-1]:
            for l_i in range(len(perc.layers) - 2, 0, -1):
                l = perc.layers[l_i]
                sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)

                delta = []

                for j in range(l.n_neurons):
                    # средневзвешенная delta выходов

                    delta.append(sum[j] * l.derivative(l.XW[j]))

                    for k in range(l.n_input):
                        o_k = perc.layers[l_i - 1].out[k]
                        perc.layers[l_i].w[j][k] -= l.lr * o_k * delta[j]

                lastDelta = np.array(delta)

    plt.plot(epohs, errors)
    plt.show()


def gradient2(perc):
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
            lastDelta = []
            
            grads = - np.dot(np.transpose([delta]), [perc.layers[l_i - 1].out])
            perc.layers[-1].w[j][k] += l.lr * grads

            for j in range(l.n_neurons):
                lastDelta.append(perc.dloss(y[i][j], l.out[j])) # * l.derivative(l.XW[j]))
                '''for k in range(l.n_input):
                    o_k = perc.layers[-2].out[k]'''
                    
            lastDelta = np.array(lastDelta)

            # скрытые слои
            # for l in perc.layers[1:-1:-1]:
            for l_i in range(len(perc.layers) - 2, 0, -1):
                l = perc.layers[l_i]
                sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)

                delta = []

                for j in range(l.n_neurons):
                    # средневзвешенная delta выходов

                    delta.append(sum[j] * l.derivative(l.XW[j]))

                    for k in range(l.n_input):
                        o_k = perc.layers[l_i - 1].out[k]
                        perc.layers[l_i].w[j][k] -= l.lr * o_k * delta[j]

                lastDelta = np.array(delta)

                '''grads = np.dot(np.transpose([delta]), [perc.layers[l_i - 1].out])

                if len(l.pred_grads) == 0 or len(l.pred_w) == 0:
                    l.pred_grads = np.array([[0.01 for _ in range(l.n_input)] for _ in range(l.n_neurons)])
                    l.pred_w = np.array([[0.01 for _ in range(l.n_input)] for _ in range(l.n_neurons)])
                    for i in range (l.n_neurons) :
                            for j in range (l.n_input) :
                                l.w[i][j] -= l.lr * grads[i][j]
                                l.pred_w[i][j] = -l.lr * grads[i][j]

                for i in range (l.n_neurons) :
                    for j in range (l.n_input) :
                        beta = min(max(0, np.sum(grads[i][j] * grads[i][j]) / (np.sum(l.pred_grads[i][j] * l.pred_grads[i][j]) + 1e-6)), 1)
                        delta_param = (l.pred_grads[i][j]) + beta * l.pred_w[i][j]

                        l.w[i][j] += l.lr * delta_param
                        l.pred_grads[i][j] = grads[i][j]
                        l.pred_w[i][j] = delta_param'''

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

            # последний слой
            l = perc.layers[-1]
            lastDelta = []

            for j in range(l.n_neurons):
                lastDelta.append(perc.dloss(y[i][j], l.out[j])) # * l.derivative(l.XW[j]))
                for k in range(l.n_input):
                    o_k = perc.layers[-2].out[k]
                    perc.layers[-1].w[j][k] -= l.lr * o_k * lastDelta[j]
            lastDelta = np.array(lastDelta)

            # скрытые слои
            # for l in perc.layers[1:-1:-1]:
            for l_i in range(len(perc.layers) - 2, 0, -1):
                l = perc.layers[l_i]
                sum = np.dot(np.transpose(perc.layers[l_i + 1].w), lastDelta)

                delta = []

                for j in range(l.n_neurons):
                    # средневзвешенная delta выходов

                    delta.append(sum[j] * l.derivative(l.XW[j]))

                    for k in range(l.n_input):
                        o_k = perc.layers[l_i - 1].out[k]
                        perc.layers[l_i].w[j][k] -= l.lr * o_k * delta[j]

                lastDelta = np.array(delta)

                grads = lastDelta * perc.layers[l_i - 1].out   

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
