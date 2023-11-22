def func1(x):
    return (x-2)**2 + 2

def F(p):
    a, b, f0, n = 180, 2, 15, 2
    return a*(p.x1**2 - p.x2)**2 + b*(p.x1-1)**2 + f0

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

x = (5, 2)

def F(x1, x2):
    return x1+x2
print(F(*x))

