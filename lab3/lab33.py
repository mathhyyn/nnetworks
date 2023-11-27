import numpy as np
import time

def F(x):
    a, b, f0 = 180, 2, 15
    return a*(x[0]**2 - x[1])**2 + b*(x[0]-1)**2 + f0

def dF(x):
    a, b = 180, 2
    return np.array([a*2*x[0]*(x[0]**2 - x[1]) + b*2*(x[0]-1), -a*2*(x[0]**2 - x[1])])


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

# Метод Бройдена-Флетчера-Гольдфарба-Шенно
def bfgs_method(func, grad_func, x0):
    n = len(x0)
    H = np.eye(n)
    alpha = 0.01
    max_iters = 50000
    eps1, eps2 = 1e-6, 1e-16
    prev_grad = []
    x = x0
    iter = 1

    for _ in range(max_iters):
        grad = grad_func(x)
        prev_grad = grad[:]
        prev_x = x[:]

        alpha = golden_section_search(
            lambda lr: func(x - lr * grad), 1e-6, 1e-1)

        p = -np.dot(H, grad)
        s = alpha * p
        x += s
        grad = grad_func(x)
        y = grad - prev_grad
        rho = 1 / np.dot(y, s)

        I = np.eye(n)
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H = np.dot(np.dot(A, H), B) + rho * np.outer(s, s)

        if np.linalg.norm(s) < eps1 and abs(func(x) - func(prev_x)) < eps2:
            # двукратное выполнение условия
            if second_time:
                break
            else:
                second_time = True
        else:
            second_time = False
        iter += 1

    print("Кол-во итераций:", iter)
    return x

# Метод Девидона-Флетчера-Пауэлла.
def dfp_method(func, grad_func, x0):
    n = len(x0)
    H = np.eye(n)
    alpha = 0.01
    max_iters = 10000
    eps1, eps2 = 1e-6, 1e-16
    prev_grad = []
    x = x0
    iter = 1

    for _ in range(max_iters):
        grad = grad_func(x)
        prev_grad = grad[:]
        prev_x = x[:]

        alpha = golden_section_search(
            lambda lr: func(x - lr * grad), 1e-6, 1e-1)

        p = -np.dot(H, grad)
        s = alpha * p  # dx
        x += s
        grad = grad_func(x)
        y = grad - prev_grad  # dg
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        A = np.dot(s, s.T) / np.dot(s.T, y)
        B = np.dot(np.dot(np.dot(H, y), y.T), H.T) / np.dot(np.dot(y.T, H), y)
        H += A - B

        if np.linalg.norm(s) < eps1 and abs(func(x) - func(prev_x)) < eps2:
            # двукратное выполнение условия
            if second_time:
                break
            else:
                second_time = True
        else:
            second_time = False
        iter += 1

    print("Кол-во итераций:", iter)
    return x


start_time = time.time()
initial_x = np.array([0.0, 0.0])
print("BFGS:")
result = bfgs_method(F, dF, initial_x)
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))

start_time = time.time()
initial_x = np.array([0.0, 0.0])
print("\nDFP:")
result = dfp_method(F, dF, initial_x)
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))
