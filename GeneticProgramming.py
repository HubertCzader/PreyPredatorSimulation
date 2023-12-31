import deap.gp
from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt

from Simulation import run_simulation
import multiprocessing
from Utils import create_pset, create_stats, create_toolbox, plot_logbook, plot_tree
import numpy as np


PREY_TERMINALS = [
    'go_to_food',
    'go_from_predator',
    'do_nothing',
    'eat',
    'reproduce'
]
PREY_ARGS = [
    "food_nearby",
    "predator_nearby",
    "hunger_over_half",
    "over_reproduction_age",
    "on_grass"
]
pset_prey = create_pset(PREY_TERMINALS, PREY_ARGS)

PREDATOR_TERMINALS = [
    'go_to_prey',
    'do_nothing',
    'eat',
    'reproduce'
]
PREDATOR_ARGS = [
    "prey_nearby",
    "hunger_over_half",
    "over_reproduction_age",
    "caught_prey"
]
pset_predator = create_pset(PREDATOR_TERMINALS, PREDATOR_ARGS)

# prim_root = deap.gp.Primitive('sequence2', (object, object), object)
#
# term_eat = deap.gp.Terminal('eat', True, object)
# term_predator_nearby = deap.gp.Terminal('predator_nearby', True, object)
#
# best_prey = [prim_root, term_predator_nearby, term_eat]\'eat\'

# With grass
# best_prey = 'selector2(selector3(sequence2(predator_nearby, \'go_from_predator\'), sequence2(hunger_over_half, \'reproduce\'), sequence2(on_grass, \'eat\')), \'go_to_food\')'
# best_predator = 'selector3(sequence2(hunger_over_half, \'reproduce\'), sequence2(caught_prey, \'eat\'), \'go_to_prey\')'

best_prey = 'selector2(sequence2(predator_nearby, \'go_from_predator\'), \'reproduce\')'
best_predator = 'selector3(\'reproduce\', sequence2(caught_prey, \'eat\'), \'go_to_prey\')'

pop_prey = None
pop_predator = None

def eval_prey(individual):
    routine = gp.compile(individual, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    _, res = run_simulation(routine, predator_routine)
    return res,

def eval_predator(individual):
    routine = gp.compile(individual, pset_predator)
    prey_routine = gp.compile(best_prey, pset_prey)
    res, _ = run_simulation(prey_routine, routine)
    return res,

def lotka_volterra(x, y, length):
    xs = [x]  # preys
    ys = [y]  # predators

    # alpha = 0.25 * ((8 / 25) / 4 - 1 / 16)
    # beta = 1 / 50 ** 3
    # gamma = 1 / 50 ** 3
    # delta = 1 / 256

    alpha = 0.03  # natural growth rate of preys
    beta = 0.001  # death rate per encounter of preys due to predation
    gamma = 0.001  # natural growth rate of predators
    delta = 0.05  # natural death rate of predators in the absence of food

    timestep = 0.01

    steps = int(length / timestep)
    inverse_timestep = int(1 / timestep)

    for i in range(steps):
        dx = alpha * x - beta * x * y
        dy = gamma * x * y - delta * y

        # Calculating next element by Euler's method
        x += dx * timestep
        y += dy * timestep

        if (i + 1) % inverse_timestep == 0 and len(xs) < length:
            xs.append(x)
            ys.append(y)

    return xs, ys

def eval_prey_lv(individual):
    routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    pred_counts, prey_counts = run_simulation(routine, predator_routine, lotka_volterra=True)
    correct_prey, correct_pred = lotka_volterra(prey_counts[0], pred_counts[0], len(prey_counts))

    # print(prey_counts)

    plt.plot(range(len(pred_counts)), pred_counts, label='predator')
    plt.plot(range(len(prey_counts)), prey_counts, label='prey')
    plt.legend()
    plt.show()

    mse = (np.square(np.array(prey_counts) - np.array(correct_prey))).mean() + \
          (np.square(np.array(pred_counts) - np.array(correct_pred))).mean()\

    return -mse,

def eval_predator_lv(individual):
    routine = gp.compile(individual, pset_predator)
    prey_routine = gp.compile(best_prey, pset_prey)
    prey_counts, pred_counts = run_simulation(prey_routine, routine, lotka_volterra=True)
    correct_prey, correct_pred = lotka_volterra(prey_counts[0], pred_counts[0], len(prey_counts))

    mse = (np.square(np.array(prey_counts) - np.array(correct_prey))).mean() + \
          (np.square(np.array(pred_counts) - np.array(correct_pred))).mean()\

    return -mse,

def show_behaviour(lotka_volterra: bool, draw_grid: bool):
    prey_routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    prey_counts, pred_counts = run_simulation(prey_routine, predator_routine, lotka_volterra=lotka_volterra,
                                              draw_grid=draw_grid)
    return prey_counts, pred_counts

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

if __name__ == '__main__':
    prey_logs = []
    predator_logs = []
    prey_counts, pred_counts = show_behaviour(lotka_volterra=True, draw_grid=False)
    # correct_prey, correct_pred = lotka_volterra(prey_counts[0], pred_counts[0], len(prey_counts))
    plt.figure()
    plt.plot(range(len(prey_counts)), prey_counts, label='Preys')
    plt.plot(range(len(pred_counts)), pred_counts, label='Predators')
    plt.title("Population of preys and predators over time")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.show()