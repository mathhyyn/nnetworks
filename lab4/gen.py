from main import createPerc, plt

perc1 = createPerc()

import numpy as np
import time
import random

class Individ:
    def __init__(self, layers_num, neuron_num):
        self.layers_num = layers_num
        self.neuron_num = neuron_num
        self.params = [layers_num] + neuron_num
        self.p = createPerc(layers_num, neuron_num)
        self.p.NAG(log=False)

def createIndivid(params):
    layers_num = int(params[0])
    neuron_num = params[1:]
    neuron_num = neuron_num[:layers_num] + [10] * (layers_num - len(neuron_num))
    return Individ(layers_num, neuron_num)

def generate_perc_population(size):
    population = []
    for _ in range(size):
        layers_num = random.randint(1, 10)
        neuron_num = []
        for _ in range(layers_num):
            neuron_num.append(random.randint(1, 30))
        p = Individ(layers_num, neuron_num)
        population.append(p)
    return population


def F(x):
    return x.p.checkCorrectness(log = False)

# Определение функции фитнеса
def fitness_function(x):
    return 1 / F(x)

# Селекция
def selection(population, fitness_func, retain_ratio=0.5):
    fitness_scores = [(fitness_func(ind), ind) for ind in population]
    sorted_population = [ind for _, ind in sorted(fitness_scores, key=lambda x: x[0])]
    retain_length = int(len(sorted_population) * retain_ratio)
    retain_length = 2 if retain_length < 2 else retain_length
    parents = sorted_population[:retain_length]
    return parents

# Скрещивание
def crossover(parents):
    parents = [p.params for p in parents]
    children = []
    while len(children) < len(parents):
        id1 = random.randint(0, len(parents)-1)
        id2 = random.randint(0, len(parents)-1)
        if id1 != id2:
            parent1 = parents[id1]
            parent2 = parents[id2]
            maxlen = min(len(parent1), len(parent2))
            crossover_point = random.randint(0, maxlen)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            children.extend([createIndivid(child1), createIndivid(child2)])
    return children

# Мутация
def mutate(params):
    rand_index = random.randint(0, len(params)-1)
    params[rand_index] += random.randint(-2, 2)
    params[rand_index] = max(1, params[rand_index])
    return createIndivid(params)

def mutation(children, mutation_chance=0.2):
    for child in children:
        if random.random() < mutation_chance:
            child = mutate(child.params)

    return children

xs = []
ys = []

# Генетический алгоритм
def genetic_algorithm(population_size, generations, mutation_rate, crossover_rate):
    population = generate_perc_population(size=population_size)
    for i in range(generations):
        print(i)
        parents = selection(population, fitness_function, crossover_rate)
        offspring = crossover(parents)
        offspring = mutation(offspring, mutation_rate)
        population = parents + offspring
        result = population[np.argmin([fitness_function(x) for x in population])]
        xs.append(i)
        ys.append(F(result))
    
    for p in population:
        print("Слои:", p.layers_num, "Кол-во нейронов:", p.neuron_num)
        print("Результат:", F(p))
    best_solution = population[np.argmin([fitness_function(x) for x in population])]
    return best_solution

start_time = time.time()
print("Генетический алгоритм:")
result = genetic_algorithm(population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.5)
print("Время выполнения:", time.time() - start_time, "c")
plt.plot(xs, ys)
plt.show()
print("Кол-во скрытых слоев:", result.layers_num, "Кол-во нейронов на скрытых слоях:", result.neuron_num)
print("Лучший результат:", F(result))
print()