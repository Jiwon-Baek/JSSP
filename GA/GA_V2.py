import pygad
import numpy as np
from GA import *


function_inputs = [0.1 * i for i in range(100)]
desired_output = 0

def get_sorted_indices(lst):
    return sorted(range(len(lst)), key=lambda k: lst[k])

def fitness_func(ga_instance, solution, solution_idx):
    # global ga_instance
    # print("generations_completed", ga_instance.generations_completed)

    seq = get_sorted_indices(solution)
    ind = Individual(seq)
    fitness = 10000.0 / ((ind.makespan - 1233)**2)
    return fitness

fitness_function = fitness_func

RANDOM_SEED = 2
num_generations = 10
num_parents_mating = 20

sol_per_pop = 100
num_genes = len(function_inputs)

init_range_low = 0
init_range_high = 100

# parent_selection_type = "sss"
parent_selection_type = "rws"
# parent_selection_type = "tournament"
keep_parents = 10 # keep all parents : -1
keep_elitism = 2
crossover_type = "single_point"
# crossover_type = "two_points"

# mutation_type = "random"
mutation_type = "swap"
# mutation_type = "inversion"
# mutation_type = "scramble"

mutation_percent_genes = 20

ga_instance = pygad.GA(random_seed=RANDOM_SEED,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       keep_elitism=keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       save_solutions=True,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()

seq = get_sorted_indices(solution)
ind = Individual(seq)
prediction = ind.makespan
print("Parameters of the best solution : {solution}".format(solution=ind.machine_seq))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

ga_instance.plot_fitness()
ga_instance.plot_new_solution_rate()


"""
makespan 1295

fitness = 10000.0 / ((ind.makespan - 1233)**2)
RANDOM_SEED = 2
num_generations = 100
num_parents_mating = 20
sol_per_pop = 100
num_genes = len(function_inputs)
init_range_low = 0
init_range_high = 100
parent_selection_type = "rws"
keep_parents = 10 # keep all parents : -1
keep_elitism = 2
crossover_type = "single_point"
mutation_type = "swap"
mutation_percent_genes = 50
"""



