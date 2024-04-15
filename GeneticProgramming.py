from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt

from Simulation import run_simulation
from LotkaVoltera import lotka_volterra
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


best_prey = 'selector2(sequence2(predator_nearby, \'go_from_predator\'), \'reproduce\')'
best_predator = 'selector3(\'reproduce\', sequence2(caught_prey, \'eat\'), \'go_to_prey\')'

pop_prey = None
pop_predator = None

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


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


def eval_prey_lv(individual):
    routine = gp.compile(individual, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    predators, preys = run_simulation(routine, predator_routine, lotka_volterra=True)
    expected_preys, expected_predators = lotka_volterra(preys[0], predators[0], len(preys))

    plt.plot(range(len(predators)), predators, label='predator')
    plt.plot(range(len(preys)), preys, label='prey')
    plt.legend()
    plt.show()

    mse = (np.square(np.array(preys) - np.array(expected_preys))).mean() + \
          (np.square(np.array(predators) - np.array(expected_predators))).mean()

    return -mse,


def eval_predator_lv(individual):
    routine = gp.compile(individual, pset_predator)
    prey_routine = gp.compile(best_prey, pset_prey)
    preys, predators = run_simulation(prey_routine, routine, lotka_volterra=True)
    expected_preys, expected_predators = lotka_volterra(preys[0], predators[0], len(preys))

    mse = (np.square(np.array(preys) - np.array(expected_preys))).mean() + \
          (np.square(np.array(predators) - np.array(expected_predators))).mean()

    return -mse,


def show_behaviour(lotka_voltera_model: bool, draw_grid: bool):
    prey_routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    preys, predators = run_simulation(prey_routine, predator_routine, lotka_volterra=lotka_voltera_model,
                                      draw_grid=draw_grid)
    return preys, predators


if __name__ == '__main__':
    prey_logs = []
    predator_logs = []
    prey_counts, pred_counts = show_behaviour(lotka_voltera_model=True, draw_grid=False)
    correct_prey, correct_pred = lotka_volterra(prey_counts[0], pred_counts[0], len(prey_counts))

    plt.figure()
    plt.plot(range(len(prey_counts)), prey_counts, label='Preys')
    plt.plot(range(len(pred_counts)), pred_counts, label='Predators')
    plt.title("Population of preys and predators over time")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
