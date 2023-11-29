import numpy as np
import time

def F(x):
    a, b, f0 = 180, 2, 15
    return a*(x[0]**2 - x[1])**2 + b*(x[0]-1)**2 + f0

def dF(x):
    a, b = 180, 2
    return np.array([a*2*x[0]*(x[0]**2 - x[1]) + b*2*(x[0]-1), -a*2*(x[0]**2 - x[1])])

# Матрица Якоби (производных)
def jacobian(x):
    return np.array([[2*x[0], 0], [0, 2*x[1]]])


# Метод Левенберга-Марквардта
def levenberg_marquardt(func, gradient, x0, lamda=1):
    n = len(x0)
    max_iters = 10000
    eps1, eps2 = 1e-6, 1e-16
    alpha = 1
    x = x0
    iter = 1

    for _ in range(max_iters):
        grad = gradient(x)
        jac = jacobian(x)
        hessian = np.dot(jac.T, jac) + alpha * np.eye(n)
        step = np.linalg.solve(hessian, -grad)
        new_x = x + step
        if np.linalg.norm(step) < eps1 and abs(func(x) - func(new_x)) < eps2:
            break
        
        if func(new_x) < func(x):
            alpha /= 2
            x = new_x
        else:
            alpha *= 2
        iter += 1

    print("Кол-во итераций:", iter)
    return x

start_time = time.time()
initial_x = np.array([0.0, 0.0])
print("Метод Левенберга-Марквардта:")
result = levenberg_marquardt(F, dF, initial_x)
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))