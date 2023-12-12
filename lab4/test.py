import numpy as np

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

# Пример использования
# Замените функцию gradient на вашу собственную функцию градиента
def gradient(x):
    return np.array([2 * (x[0]-1), 2 * x[1]])  # Пример квадратичной функции

initial_position = np.array([1.0, 1.0])  # Начальная позиция
learning_rate = 0.1  # Скорость обучения
momentum = 0.9  # Момент
max_iterations = 1000  # Максимальное количество итераций
tolerance = 1e-6  # Порог сходимости

result = NAG(gradient, initial_position, learning_rate, momentum, max_iterations, tolerance)
print("Результат оптимизации:", result)
