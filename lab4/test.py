import numpy as np

print(np.sqrt(np.array([[1, 2, 3], [4, -2, -3]]) ** 2))

def NAG(gradient, initial_position, lr, momentum, max_iterations, tol):
    position = initial_position
    vt = np.zeros_like(position)
    
    for i in range(max_iterations):        
        # Вычисляем "псевдо-градиент" с применением момента
        pseudo_gradient = gradient(position - momentum * vt)
        
        # Обновляем скорость
        vt += lr * pseudo_gradient
        
        # Обновляем позицию
        position -= vt
    
    return position

import numpy as np

def adam_optimizer(alpha=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, max_iterations=1000):
    # Инициализация параметров
    theta = [0, 1] # Начальные параметры
    m = np.zeros_like(theta)  # Инициализация момента первого порядка
    v = np.zeros_like(theta)  # Инициализация момента второго порядка
    t = 0  # Итерационный шаг

    while t < max_iterations:
        t += 1
        # Вычисляем градиент функции потерь по параметрам
        gradients = gradient(theta)
        # Обновляем первый момент
        m = beta1 * m + (1 - beta1) * gradients
        # Обновляем второй момент
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        # Смещение моментов
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        # Обновляем параметры
        theta = theta - alpha * m_hat / (np.sqrt(v_hat) + epsilon)

    return theta

# Пример использования
# Замените функцию gradient на вашу собственную функцию градиента
def gradient(x):
    return np.array([2 * (x[0]-1), 2 * x[1]])  # Пример квадратичной функции

initial_position = np.array([1.0, 1.0])  # Начальная позиция
learning_rate = 0.1  # Скорость обучения
momentum = 0.9  # Момент
max_iterations = 1000  # Максимальное количество итераций
tolerance = 1e-6  # Порог сходимости

#result = NAG(gradient, initial_position, learning_rate, momentum, max_iterations, tolerance)
result = adam_optimizer()
print("Результат оптимизации:", result)
