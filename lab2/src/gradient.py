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
                        try:
                            perc.layers[l_i].w[j][k] -= l.lr * o_k * delta[j]
                        except:
                            print(o_k, delta[j])

                lastDelta = np.array(delta)

    plt.plot(epohs, errors)
    plt.show()
