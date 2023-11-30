import numpy as np
import time
import random
import matplotlib.pyplot as plt

def F(x):
    a, b, f0 = 180, 2, 15
    return a*(x[0]**2 - x[1])**2 + b*(x[0]-1)**2 + f0

# Определение функции фитнеса
def fitness_function(x):
    return 1 / F(x)

# Селекция
def selection(population, fitness_func, retain_ratio=0.5):
    fitness_scores = {tuple(ind): fitness_func(ind) for ind in population}
    sorted_population = [list(ind) for ind in sorted(fitness_scores, key=fitness_scores.get, reverse=True)]
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
            child[random.randint(0, len(child)-1)] += np.random.uniform(-0.5, 0.5)
    return children

xs = []
ys = []

# Генетический алгоритм
def genetic_algorithm(population_size, dimension, generations, mutation_rate, crossover_rate):
    population = np.random.uniform(low=-5, high=5, size=(population_size, dimension))
    for i in range(generations):
        parents = selection(population, fitness_function, crossover_rate)
        offspring = crossover(parents)
        offspring = mutation(offspring, mutation_rate)
        population = parents + offspring
        result = population[np.argmax([fitness_function(x) for x in population])]
        xs.append(i)
        ys.append(F(result))
    
    best_solution = population[np.argmax([fitness_function(x) for x in population])]
    return best_solution

start_time = time.time()
print("Генетический алгоритм:")
result = genetic_algorithm(population_size=60, dimension=2, generations=50, mutation_rate=0.15, crossover_rate=0.5)
plt.plot(xs, ys)
plt.show()
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))