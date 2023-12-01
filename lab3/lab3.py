import numpy as np
import time
import matplotlib.pyplot as plt

def F(x):
    a, b, f0 = 180, 2, 15
    return sum(a*(x[i]**2 - x[i+1])**2 + b*(x[i]-1)**2 for i in range(len(x)-1)) + f0

def dF(x):
    a, b = 180, 2
    return np.array([a*2*2*x[0]*(x[0]**2 - x[1]) + b*2*(x[0]-1), -a*2*(x[0]**2 - x[1])])

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

xs, ys = [], []

# Метод наискорейшего градиентного спуска
def gradient_descent(func, grad_func, x0):
    alpha = 0.01
    max_iters = 20000
    eps1, eps2 = 1e-6, 1e-16
    prev_x = x0.copy()
    x = x0
    iter = 1
    second_time = False
    for i in range(max_iters):
        grad = grad_func(x)
        alpha = golden_section_search(lambda lr: func(x - lr * grad), 1e-5, 1e-1)
        prev_x = x.copy()
        x -= alpha * grad
        if np.linalg.norm(x - prev_x) < eps1 and abs(func(x) - func(prev_x)) < eps2:
            # двукратное выполнение условия
            if second_time:
                break
            else:
                second_time = True
        else:
            second_time = False
        iter += 1
        xs.append(i)
        ys.append(F(x))
    print("Кол-во итераций:", iter)
    return x



# Метод Флетчера-Ривза
def fletcher_reeves(func, grad_f, x0):
    alpha = 0.01
    max_iters = 1000
    eps1, eps2, eps3 = 1e-6, 1e-16, 1e-16
    prev_x = x0.copy()
    x = x0
    prev_grad = []
    d = -grad_f(x)
    iter = 1
    second_time = False

    for i in range(max_iters):
        grad = grad_f(x)
        alpha = golden_section_search(
            lambda lr: func(x + lr * d), 1e-6, 1e-3)
        if len(prev_grad) != 0:
            beta = np.dot(grad, grad) / np.dot(prev_grad, prev_grad)
            d = -grad + beta * d
        prev_x = x.copy()
        prev_grad = grad.copy()
        x += alpha * d

        if np.linalg.norm(x - prev_x) < eps1 and abs(func(x) - func(prev_x)) < eps2 or np.linalg.norm(grad) < eps3:
            # двукратное выполнение условия
            if second_time:
                break
            else:
                second_time = True
        else:
            second_time = False
        iter += 1
        xs.append(i)
        ys.append(F(x))

    print("Кол-во итераций:", iter)
    return x

# Метод Полака-Рибьера
def polak_ribiere(func, grad_f, x0):
    alpha = 0.01
    max_iters = 10000
    eps1, eps2, eps3 = 1e-6, 1e-16, 1e-16
    prev_x = x0.copy()
    x = x0
    prev_grad = []
    d = -grad_f(x)
    iter = 1
    n = 5
    second_time = False

    for i in range(max_iters):
        grad = grad_f(x)
        # выполненеие на каждом n-ом шаге итерации наискорейшего спуска
        if i % n != 0:
            beta = np.dot(grad.T, grad - prev_grad) / np.dot(prev_grad, prev_grad)
            d = -grad + beta * d
        prev_x = x.copy()
        prev_grad = grad.copy()
        if i % n != 0:
            alpha = golden_section_search(
                lambda lr: func(x + lr * d), 0, 1e-3)
            x += alpha * d
        else:
            alpha = golden_section_search(
                lambda lr: func(x - lr * grad), 0, 1e-3)
            x -= alpha * grad

        iter += 1

        if np.linalg.norm(x - prev_x) < eps1 and abs(func(x) - func(prev_x)) < eps2 or np.linalg.norm(grad) < eps3:
            # двукратное выполнение условия
            if second_time:
                break
            else:
                second_time = True
        else:
            second_time = False
        xs.append(i)
        ys.append(F(x))

    print("Кол-во итераций:", iter)
    return x


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

    for i in range(max_iters):
        grad = grad_func(x)
        prev_grad = grad.copy()
        prev_x = x.copy()

        alpha = golden_section_search(
            lambda lr: func(x - lr * grad), 1e-6, 1e-3)

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
        xs.append(i)
        ys.append(F(x))

    print("Кол-во итераций:", iter)
    return x

# Метод Девидона-Флетчера-Пауэлла.
def dfp_method(func, grad_func, x0):
    n = len(x0)
    H = np.eye(n)
    alpha = 0.01
    max_iters = 20000
    eps1, eps2 = 1e-6, 1e-16
    prev_grad = []
    x = x0
    iter = 1

    for i in range(max_iters):
        grad = grad_func(x)
        prev_grad = grad.copy()
        prev_x = x.copy()
        
        p = -np.dot(H, grad)

        alpha = golden_section_search(
            lambda lr: func(x + lr * p), 1e-6, 1e-3)

        
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
        xs.append(i)
        ys.append(F(x))

    print("Кол-во итераций:", iter)
    return x





# Метод Левенберга-Марквардта
def levenberg_marquardt(func, gradient, x0, lamda=1):
    def jacobian(x):
        a, b = 180, 2
        return np.array([[a*2*2*x[0]*(x[0]**2 - x[1]) + b*2*(x[0]-1), 0], [0, -a*2*(x[0]**2 - x[1])]])
    n = len(x0)
    max_iters = 10000
    eps1, eps2 = 1e-6, 1e-16
    alpha = 1
    x = x0
    iter = 1

    for i in range(max_iters):
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
        xs.append(i)
        ys.append(F(x))

    print("Кол-во итераций:", iter)
    return x


methods = [{'name':"Метод наискорейшего градиентного спуска", 'func': gradient_descent, 'x0': np.array([0.0, 0.0])}, 
           {'name':"Метод Флетчера-Ривза", 'func': fletcher_reeves, 'x0': np.array([2.0, 0.0])},
           {'name':"Метод Полака-Рибьера", 'func': polak_ribiere, 'x0': np.array([2.0, 0.0])},
           {'name':"BFGS", 'func': bfgs_method, 'x0': np.array([0.0, 0.0])}, 
           {'name':"DFP", 'func': dfp_method, 'x0': np.array([0.0, 0.0])},
           {'name':"Метод Левенберга-Марквардта", 'func': levenberg_marquardt, 'x0': np.array([0.0, 0.0])}
           ]

for method in methods:
    xs, ys = [], []
    start_time = time.time()
    print(f"\n{method['name']}:")
    result = method['func'](F, dF, method['x0'])
    print("Время выполнения:", time.time() - start_time, "c")
    print("Точка минимума функции:", result)
    print("Минимум функции:", F(result))
    plt.plot(xs[:1000], ys[:1000], label=method['name'])

plt.legend()
plt.show()


