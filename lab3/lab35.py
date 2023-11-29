import numpy as np
import time

def F(x):
    a, b, f0 = 180, 2, 15
    return a*(x[0]**2 - x[1])**2 + b*(x[0]-1)**2 + f0

# Определение функции фитнеса
def fitness_function(x):
    return 1 / F(x)

# Выбор родителей
def select_parents(population, fitness_values):
    a = np.argmax(fitness_values)
    parent1 = population[a]
    fitness_values[a] = -1
    parent2 = population[np.argmax(fitness_values)]
    return parent1, parent2

# Оператор скрещивания (одноточечный)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

# Оператор мутации
def mutate(child, mutation_rate):
    for i in range(len(child)):
        if mutation_rate[0] < np.random.rand() < mutation_rate[1]:
            child[i] += np.random.uniform(-0.5, 0.5)
    return child

# Генетический алгоритм
def genetic_algorithm(population_size, dimension, generations, mutation_rate, crossover_rate):
    population = np.random.uniform(low=-5, high=5, size=(population_size, dimension))
    for _ in range(generations):
        fitness_values = [fitness_function(x) for x in population]
        new_population = []

        for _ in range(population_size):
            parent1, parent2 = select_parents(population, fitness_values)
            if crossover_rate[0] < np.random.rand() < crossover_rate[1]:
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, mutation_rate)
                child2 = mutate(child2, mutation_rate)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        population = np.array(new_population)
    
    best_solution = population[np.argmax([fitness_function(x) for x in population])]
    return best_solution

start_time = time.time()
print("Генетический алгоритм:")
result = genetic_algorithm(population_size=200, dimension=2, generations=500, mutation_rate=[0.05, 0.2], crossover_rate=[0.3, 0.5])
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))
