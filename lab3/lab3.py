from scipy.optimize import minimize_scalar, minimize
def func1(x):
    return (x-2)**2 + 2

def F(x):
    a, b, f0 = 180, 2, 15
    return a*(x[0]**2 - x[1])**2 + b*(x[0]-1)**2 + f0

# Золотое сечение
def golden_section_search(f, a, b, tol=1e-5):
    gr = (5 ** 0.5 - 1) / 2  
    x1 = b - (b - a) * gr
    x2 = a + (b - a) * gr
    while abs(x1 - x2) > tol:
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        x1 = b - (b - a) * gr
        x2 = a + (b - a) * gr
    return (b + a) / 2

print(golden_section_search(func1, -5, 5))
print(minimize_scalar(func1))

x = (5, 2)

def F(x1, x2):
    return x1+x2
print(F(*x))

import numpy as np

def bfgs_method(f, grad_f, x0, max_iter=1000, tol=1e-5):
    n = len(x0)
    H = np.eye(n)  # Начальное приближение обратной гессианы
    x = [1, 1, 1]
    for _ in range(max_iter):
        gradient = grad_f(x)
        if np.linalg.norm(gradient) < tol:
            print("BREAK")
            break  # Прекращаем, если градиент стал достаточно мал
        p = - np.dot(H, gradient)  # Направление спуска
        '''alpha = 1.0  # Вычисление шага с помощью линейного поиска
        while f(x + alpha * p) > f(x) + 0.5 * alpha * np.dot(gradient, p):
            alpha *= 0.5'''
        I = np.eye(n)
        s = 0.01 * p
        x_next = x + s
        y = grad_f(x_next) - gradient
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)
        rho_ = np.dot(y.T, s)
        rho = 1 if rho_ == 0 else 1 / rho_
        A = I - rho * np.dot(s, y.T)
        B = I - rho * np.dot(y, s.T)
        H = np.dot(np.dot(A, H), B) + rho * np.dot(s, s.T)
        x = x_next[:]
    print(H)
    return x

# Пример использования
def my_function(x):
    return x[0]**2 + x[1]**2 + (x[2] - 3)**2

def my_gradient(x):
    return np.array([2*x[0], 2*x[1], 2*(x[2]-3)])

x0 = np.array([1, 1, 1])
result = bfgs_method(my_function, my_gradient, x0)
print(result)
print(minimize(my_function, np.array([1, 1, 1])))

print(np.array([[1,2], [3,4]]).flatten())
print(np.array([[1,2], [3,4]]).T)


