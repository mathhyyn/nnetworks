import numpy as np
import time

def F(x):
    a, b, f0 = 180, 2, 15
    return a * (x[0] ** 2 - x[1]) ** 2 + b * (x[0] - 1) ** 2 + f0

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
    c = np.random.rand()
    child1 = c * parent1 + (1 - c) * parent2
    child2 = (1 - c) * parent1 + c * parent2
    return child1, child2

# Оператор мутации
def mutate(population, mutation_rate):
    mutants = []
    for p in population:
        if np.random.rand() < mutation_rate:
            p[np.random.randint(0, len(p) - 1)] += np.random.uniform(-0.5, 0.5)
            mutants.append(p)
    # print(mutants, len(population))
    return mutants


def is_acceptable(child):
    for c in child:
        if not (-5 < c < 5):
            return False
    return True


# проверка на допустимость потомков
def accepts(parents, children):
    res = []
    for i in range(len(children)):
        res.append(children[i] if is_acceptable(children[i]) else parents[i])
    return res


# Генетический алгоритм
def genetic_algorithm(population_size, dimension, generations, mutation_rate, crossover_rate):
    population = np.random.uniform(low=-5, high=5, size=(population_size, dimension))
    for _ in range(generations):
        fitness_values = [fitness_function(x) for x in population]
        parent1, parent2 = select_parents(population, fitness_values)
        new_population = []

        # кроссинговер (скрещивание)
        for i in range(len(population) // 2):
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
                new_population.extend(accepts([parent1, parent2], [child1, child2]))
            else:
                new_population.extend([parent1, parent2])

        # мутация
        mutants = mutate(new_population, mutation_rate)
        # Формирование новой популяции.
        new_population[np.argmin(fitness_function(x) for x in new_population)] = mutants[np.random.randint(0, len(mutants) - 1)]

        population = np.array(new_population)

    best_solution = population[np.argmax(fitness_function(x) for x in population)]
    return best_solution


start_time = time.time()
print("Генетический алгоритм:")
result = genetic_algorithm(
    population_size=60,
    dimension=2,
    generations=100,
    mutation_rate=0.15,
    crossover_rate=0.5,
)
print("Время выполнения:", time.time() - start_time, "c")
print("Точка минимума функции:", result)
print("Минимум функции:", F(result))
