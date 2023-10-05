import numpy as np
import matplotlib.pyplot as plt

def find_answ(res):
  num = 0
  min = 1
  for i in range(len(res)):
    if abs(1 - res[i]) < min:
      min = abs(1 - res[i])
      num = i
  return num

def phi(x):
  return 0
def derivative(x):
  delta = 0.001
  return (phi(x + delta) - phi(x)) / delta


def relu(x):
  return x if x > 0 else 0
def drelu(x):
  return 1 if x > 0 else 0

phi = relu
lr = 0.01


def lost(y0, y):
  return 1 / 2 * (y0 - y)**2

def dlost(y0, y):
  return y0 - y




def gradient1(perc):
  #out = perc.forward()
  x = perc.x_train
  y = perc.y_train

  num = len(x)
  
  epohs = []
  errors = []
    

  def maxErr():
    max = 0
    for i in range(num): # обучение на n тестах
      e = 0
      res = perc.forward(x[i])
      for j in range(10): # 10 нейронов
        e += lost(y[i][j], res[j])
      e /= 10 # среднее
      max = e if e > max else max
    return max

  '''def sum(x, w):
    res = 0
    for i in range(n_pixels):
      res += x[i]*w[i]
    return res'''
  
  '''def countLastDelta(y0, out, XW):
    res = dlost(y0, out) * derivative(XW) 
    return res
  
  def Delta(y0, x, w):
    XW = sum(x, w)
    y = phi(XW)
    res = dlost(y0, y) * derivative(XW) 
    return res'''

  '''def study2(x1, y1, j):
    delta = lastDelta(x1, w[j], y1)
    for i in range(n_pixels):
      w[j][i] += lr * x[i] * delta'''

  
  for step in range(100):
    if step % 5 == 0:
      print(step)
      epohs.append(step)
      errors.append(maxErr())

    for i in range(num): # обучение на n тестах
      perc.layers[0].out = x[i]

      out = perc.forward(x[i])

      # последний слой
      l = perc.layers[-1]
      phi = l.activation
      lastDelta = np.zeros(l.n_neurons)

      if len(perc.layers) > 1:
        for j in range(l.n_neurons):
          lastDelta[j] = dlost(y[i][j], l.out[j]) * derivative(l.XW[j])
          for k in range(l.n_input):
            l.w[j][k] += l.lr * perc.layers[len(perc.layers) - 2].out[k] * lastDelta[j]
      perc.lastDelta = lastDelta

      # скрытые слои
      #for l in reversed(perc.layers[1:-1]):
      for l_i in range(len(perc.layers)-2, 0, -1):
        l = perc.layers[l_i]
        phi = l.activation

        for j in range(l.n_neurons):
          delta = np.dot(lastDelta, perc.layers[-1].w[j]) * derivative(l.XW[j])
          for k in range(l.n_input):
            #print(perc.layers[len(perc.layers) - 2].out[k])
            l.w[j][k] += l.lr * perc.layers[l_i - 1].out[k] * delta

        '''
          l.w[j] = study2(l, l.w[j])
        # # 10 нейронов
          study2(x[i], y[i][j], j)
          '''

  
  #study()

  # проверка ответа
  for i in range(num):
    res = [0] * 10
    mas = [0] * 10

    res = perc.forward(x[i])

    for j in range(10):
      mas[j] = round(res[j], 2)

    output = find_answ(res)
    print(mas)
    print(y[i])
    print(find_answ(y[i]), '--->', output, '\n')

  plt.plot(epohs, errors)
  plt.show()
  
