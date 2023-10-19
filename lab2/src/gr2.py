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
        e += perc.loss(y[i][j], res[j])
      e /= 10 # среднее
      max = e if e > max else max
    return max

  
  for step in range(3):
    if step % 2 == 0:
      print(step)
      epohs.append(step)
      errors.append(maxErr())

    for i in range(num): # обучение на n тестах
      perc.layers[0].out = x[i][:]

      out = perc.forward(x[i])

      # последний слой
      l = perc.layers[-1]
      lastDelta = []
      #prevW = l.w

      for j in range(l.n_neurons):
        lastDelta.append(perc.dloss(y[i][j], l.out[j]) * l.derivative(l.XW[j]))
        for k in range(l.n_input):
          o_k = perc.layers[-2].out[k]
          perc.layers[-1].w[j][k] += l.lr * o_k * lastDelta[j]
      lastDelta = np.array(lastDelta[:])
      

      # скрытые слои
      #for l in perc.layers[1:-1:-1]:
      for l_i in range(len(perc.layers)-2, 0, -1):
        l = perc.layers[l_i]

        #sum = np.dot(np.transpose(prevW), lastDelta)
        sum = np.dot(np.transpose(perc.layers[l_i+1].w), lastDelta)

        delta = []
        #prevW = l.w
        
        for j in range(l.n_neurons):

          # средневзвешенная delta выходов
          
          '''for n_i in range(perc.layers[l_i+1].n_neurons):
            sum += lastDelta[n_i] * perc.layers[l_i+1].w[n_i][j]'''
          delta.append(sum[j] * l.derivative(l.XW[j]))

          for k in range(l.n_input):
            o_k = perc.layers[l_i - 1].out[k]
            perc.layers[l_i].w[j][k] += l.lr * o_k * delta[j]

        lastDelta = np.array(delta[:])
        
        

        '''
          l.w[j] = study2(l, l.w[j])
        # # 10 нейронов
          study2(x[i], y[i][j], j)
          '''

  
  #study()

  correct_num = 0
  # проверка ответа
  for i in range(num):
    res = [0] * 10
    mas = [0] * 10

    res = perc.forward(x[i])

    for j in range(10):
      mas[j] = round(res[j], 2)

    predicted = find_answ(res)
    print(mas)
    print(y[i])
    expected = find_answ(y[i])
    
    print(expected, '--->', predicted, '\n')
    if expected == predicted:
      correct_num += 1
  
  print(correct_num / num * 100, "%  \correctness")

  plt.plot(epohs, errors)
  plt.show()
  
