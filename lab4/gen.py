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

def generate_perc_population(size):
    population = []
    for _ in range(size):
        layers_num = random.randint(0, 10)
        first_neuron_num = random.randint(1, 30)
        neuron_num = [10]*layers_num
        if layers_num != 0:
            neuron_num[0] = first_neuron_num
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
    fitness_scores = {tuple(ind): fitness_func(ind) for ind in population}
    sorted_population = [list(ind) for ind in sorted(fitness_scores, key=fitness_scores.get)]
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
def genetic_algorithm(population_size, generations, mutation_rate, crossover_rate):
    population = generate_perc_population(size=population_size)
    '''for i in range(generations):
        parents = selection(population, fitness_function, crossover_rate)
        offspring = crossover(parents)
        offspring = mutation(offspring, mutation_rate)
        population = parents + offspring
        result = population[np.argmin([fitness_function(x) for x in population])]
        xs.append(i)
        ys.append(F(result))'''
    
    for p in population:
        print("Слои:", p.layers_num, "Кол-во нейронов:", p.neuron_num)
        print("Результат:", F(p))
    best_solution = population[np.argmin([fitness_function(x) for x in population])]
    return best_solution

start_time = time.time()
print("Генетический алгоритм:")
result = genetic_algorithm(population_size=5, generations=50, mutation_rate=0.15, crossover_rate=0.5)
print("Время выполнения:", time.time() - start_time, "c")
plt.plot(xs, ys)
plt.show()
print("Кол-во слоев:", result.layers_num, "Кол-во нейронов на скрытых слоях:", result.neuron_num)
print("Лучший результат:", F(result))