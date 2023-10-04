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
def dphi(x):
  delta = 0.001
  return (phi(x + delta) - phi(x)) / delta


def relu(x):
  return x if x > 0 else 0
def drelu(x):
  return 1 if x > 0 else 0

phi = relu
lr = 0.01


def err(y0, y):
  return 1 / 2 * (y - y0)**2








def gradient1(x, y):

  num = len(x)
  n_pixels = len(x[0])
  epohs = []
  errors = []

  w = [[0.01 for _ in range(n_pixels)] for _ in range(10)]

  def maxErr():
    max = 0
    for i in range(num): # обучение на n тестах
      e = 0
      for j in range(10): # 10 нейронов
        z = phi(sum(x[i], w[j]))
        e += err(y[i][j], z)
      e /= 10
      max = e if e > max else max
    return max

  def sum(x, w):
    res = 0
    for i in range(n_pixels):
      res += x[i]*w[i]
    return res
  
  def dErr(x, w, y0):
    XW = sum(x, w)
    y = phi(XW)
    res = [0] * len(x)
    for i in range(len(x)):
      res[i] = lr * (y0 - y) * x[i] * dphi(XW)
    return res

  def study2(x1, y1, j):
    delta = dErr(x1, w[j], y1)
    for i in range(n_pixels):
      w[j][i] += delta[i]

  def study():
    for step in range(1001):
      if step % 100 == 0:
        print(step)
        epohs.append(step)
        errors.append(maxErr())
      for i in range(num): # обучение на n тестах
        for j in range(10): # 10 нейронов
          study2(x[i], y[i][j], j)

  
  study()

  # проверка ответа
  for i in range(num):
    res = [0] * 10
    mas = [0] * 10
    for j in range(10):
      res[j] = phi(sum(x[i], w[j]))
      mas[j] = round(res[j], 2)

    output = find_answ(res)
    print(mas)
    print(y[i])
    print(find_answ(y[i]), '--->', output, '\n')

  plt.plot(epohs, errors)
  plt.show()
  
