import numpy as np

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


def fletcher_reeves(func, grad_f, x0):
    alpha = 0.001
    max_iters = 1000
    eps1, eps2 = 1e-6, 1e-16
    prev_x = x0[:]
    x = x0
    prev_grad = []
    d = -grad_f(x)
    for _ in range(max_iters):
        grad = grad_f(x)
        alpha = golden_section_search(lambda lr: func(x - lr * grad), 1e-6, 1e-3)
        if len(prev_grad) != 0:
            beta = np.dot(grad, grad) / np.dot(prev_grad, prev_grad)
            d = -grad + beta * d
        prev_x = x[:]
        prev_grad = grad[:]
        x += alpha * d
        
        if np.linalg.norm(grad) < eps1 and abs(func(x) - func(prev_x)) < eps2:
            break
    return x

initial_x = np.array([1.0, 0.0])
result = fletcher_reeves(F, dF, initial_x)
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))