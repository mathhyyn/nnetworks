import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_function(x):
    a, b, f0 = 180, 2, 15
    return sum(a*(x[i]**2 - x[i+1])**2 + b*(x[i]-1)**2 for i in range(len(x)-1)) + f0
    
def fitness_function(x):
    return 1 / rosenbrock_function(x)


import random

# Инициализация популяции
population_size = 100
population = np.random.uniform(low=-5, high=5, size=(population_size, 2))

# Селекция
def selection(population, fitness_func, retain_ratio=0.5):
    fitness_scores = {tuple(ind): fitness_func(ind) for ind in population}
    sorted_population = [list(ind) for ind in sorted(fitness_scores, key=fitness_scores.get, reverse=True)]
    #print(sorted_population)
    retain_length = int(len(sorted_population) * retain_ratio)
    retain_length = 2 if retain_length < 2 else retain_length
    parents = sorted_population[:retain_length]
    return parents

# Скрещивание
def crossover(parents):
    children = []
    while len(children) < len(parents):
        father = random.randint(0, len(parents)-1)
        mother = random.randint(0, len(parents)-1)
        if father != mother:
            parent1 = np.array(parents[father])
            parent2 = np.array(parents[mother])
            c = np.random.rand()
            child1 = c * parent1 + (1 - c) * parent2
            child2 = (1 - c) * parent1 + c * parent2
            children.extend([child1, child2])
    return children

# Мутация
def mutation(children, mutation_chance=0.2):
    for child in children:
        if random.random() < mutation_chance:
            index = random.randint(0, len(child)-1)
            child[index] += np.random.uniform(-0.5, 0.5)
    return children


generations = 100
retain_ratio = 0.5

xs = []
ys = []
result = 0

for i in range(generations):
    print(i)
    parents = selection(population, fitness_function, retain_ratio)
    offspring = crossover(parents)
    offspring = mutation(offspring)
    population = parents + offspring
    result = population[np.argmax([fitness_function(x) for x in population])]
    xs.append(i)
    ys.append(rosenbrock_function(result))
    
plt.plot(xs, ys)
plt.show()
print("Точка минимума функции:", result)
print("Минимум функции:", rosenbrock_function(result))
