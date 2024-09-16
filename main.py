import matplotlib.pyplot as plt
import numpy as np

from deap import gp, creator, base
from datetime import datetime

from Simulation import run_simulation
from Utils import create_pset, create_pset_pred


PREY_TERMINALS = [
    'go_from_predator',
    'explore',
    'reproduce',
    'do_nothing',
]
PREY_ARGS = [
    "predator_nearby",
    "last_moved",
    "over_reproduction_age",
    'idleness',
]
pset_prey = create_pset(PREY_TERMINALS, PREY_ARGS)

PREDATOR_TERMINALS = [
    'go_to_prey',
    'eat',
    'look_for_prey',
    'do_nothing',
    'reproduce',

]
PREDATOR_ARGS = [
    "prey_nearby",
    "hunger_over",
    "hungry_but_no_prey",
    "over_reproduction_age",
    "caught_prey"
]
pset_predator = create_pset_pred(PREDATOR_TERMINALS, PREDATOR_ARGS)


best_prey = 'selector4(sequence2(predator_nearby, \'go_from_predator\'), sequence2(last_moved, \'explore\'), ' \
            'sequence2(over_reproduction_age, \'reproduce\'),  \'do_nothing\')'

best_predator = 'selector3(sequence2(hunger_over, selector3(sequence2(caught_prey, \'eat\'), sequence2(prey_nearby, ' \
                '\'go_to_prey\'), \'look_for_prey\')), sequence2(over_reproduction_age, \'reproduce\'), \'do_nothing\')'

pop_prey = None
pop_predator = None

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def show_behaviour(print_state: bool, lotka_voltera_model: bool, draw_grid: bool):
    prey_routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    preys, predators = run_simulation(prey_routine, predator_routine, print_state=print_state,
                                      lotka_volterra=lotka_voltera_model, draw_grid=draw_grid)
    return preys, predators


if __name__ == '__main__':
    # prey_counts, pred_counts = show_behaviour(print_state=True, lotka_voltera_model=True, draw_grid=False)

    prey_counts, pred_counts = show_behaviour(print_state=True, lotka_voltera_model=True, draw_grid=False)
    print(prey_counts)
    print(pred_counts)

    # np.savez(f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", prey=prey_counts, pred=pred_counts)
    # plt.figure()
    # plt.plot(range(len(prey_counts)), prey_counts, label='Preys')
    # plt.plot(range(len(pred_counts)), pred_counts, label='Predators')
    # plt.title("Population of preys and predators over time")
    # plt.xlabel("Time")
    # plt.ylabel("Population")
    # plt.yticks(range(0, max(prey_counts), 100))
    # plt.legend()
    # plt.savefig(f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    # plt.show()
