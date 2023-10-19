from PIL import Image, ImageDraw #Подключим необходимые библиотеки.
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.grid()

n = 3 * 5
x = [0] * 10
#w = [[random.uniform(-1, 1) for i in range(n)] for j in range(10)]
#w = [[0.05 for i in range(n)] for j in range(10)]
y = [[0] * 10 for i in range(10)]


def getImage(N):
  image = Image.open(f"tests/{N}.png") #Открываем изображение.
  width = image.size[0] #Определяем ширину.
  height = image.size[1] #Определяем высоту.
  pix = image.load() #Выгружаем значения пикселей.

  if width != 3 and height != 5:
    sys.exit()

  x = [0] * n

  for i in range(width):
    for j in range(height):
      x[j * width + i] = 0 if pix[i, j] == (0,0,0,0) else 1

  return x

def maxErr():
  max = 0
  for i in range(10): # обучение на 10 тестах
    e = 0
    for j in range(10): # 10 нейронов
      z = phi(sum(x[i], w[j]))
      e += err(y[i][j], z)
    e /= 10
    max = e if e > max else max
  return max

def find_answ(res):
  num = 0
  min = 1
  for i in range(len(res)):
    if abs(1 - res[i]) < min:
      min = abs(1 - res[i])
      num = i
  return num

def phi():
  return 0
def dphi(x):
  delta = 0.001
  return (phi(x + delta) - phi(x)) / delta

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
  #return sigm(x) / ( 1 - sigm(x))
def th(x):
  return np.tanh(x)
  return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def dth(x):
  return 1 - th(x) ** 2


def err(y0, y):
  return 1 / 2 * (y - y0)**2

def dErr(x, w, y0):
  XW = sum(x, w)
  y = phi(XW)
  res = [0] * len(x)
  for i in range(len(x)):
    res[i] = lr * (y0 - y) * x[i] * dphi(XW)
  return res


def sum(x, w):
  res = 0
  for i in range(n):
    res += x[i]*w[i]
  return res

def study2(x, y, j):
  delta = dErr(x, w[j], y)
  for i in range(n):
    w[j][i] += delta[i]

def study():
  for step in range(501):
    if step % 100 == 0:
      #print(step)
      epohs.append(step)
      errors.append(maxErr())
    for i in range(10): # обучение на 10 тестах
      for j in range(10): # 10 нейронов
        study2(x[i], y[i][j], j)


for i in range(10):
  y[i][i] = 1
  x[i] = getImage(i)



#funcs = ['lin', 'relu', 'sigm', 'th']
funcs = ['sigm']

for func in funcs:

  phi = eval(func)
  #dphi = eval('d'+func)

  print('----------------------------------------------------------------------------')
  print(func)
  lr = 1 if func != 'lin' and func != 'th' else 0.1

  for _ in range(2):
    correct = True
    print()
    print('lr =', lr)
    epohs = []
    errors = []
    w = [[0.05 for i in range(n)] for j in range(10)]

    study()

    for i in range(10):
      res = [0] * 10
      mas = [0] * 10
      for j in range(10):
        res[j] = phi(sum(x[i], w[j]))
        mas[j] = round(res[j], 2)

      output = find_answ(res)
      print(mas)
      print(y[i])

      print(i, '--->', output, '\n')
      if output != i:
        correct = False

    if (lr == 0.01 and func != 'sigm') or (lr == 1 and func == 'sigm'):
      for j in range(10):
        print(w[j])

    plt.plot(epohs, errors, label=f'{lr} {correct}')
    lr /= 10


  plt.legend()
  plt.title(func)
  plt.show()
