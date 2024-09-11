import multiprocessing
import networkx as nx
import numpy as np

import os
import shutil

from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt
from datetime import datetime

from Simulation import run_simulation
from LotkaVoltera import lotka_volterra
from TreeUtils import simplify_tree
from Utils import create_pset, create_pset_pred, create_stats, create_toolbox, plot_logbook, plot_tree
from TreeUtils import save_tree

PREY_TERMINALS = [
    'go_from_predator',
    'go_to_food',
    'eat',
    'look_for_food',
    'reproduce',
    'do_nothing',
]

PREY_ARGS = [
    "predator_nearby",
    "food_nearby",
    "hunger_over",
    "hungry_but_no_food",
    "over_reproduction_age",
    "found_food",
]

pset_prey = create_pset(PREY_TERMINALS, PREY_ARGS)

best_prey = 'selector4(sequence2(predator_nearby, \'go_from_predator\'), sequence2(hunger_over, ' \
            'selector3(sequence2(found_food, \'eat\'), sequence2(food_nearby, \'go_to_food\'), \'look_for_food\')), ' \
            'sequence2(over_reproduction_age, \'reproduce\'), \'do_nothing\')'


PREDATOR_TERMINALS = [
    'go_to_prey',
    'eat',
    'look_for_prey',
    'reproduce',
    'do_nothing'

]
PREDATOR_ARGS = [
    "prey_nearby",
    "hunger_over",
    "hungry_but_no_prey",
    "over_reproduction_age",
    "caught_prey"
]
pset_predator = create_pset_pred(PREDATOR_TERMINALS, PREDATOR_ARGS)

best_predator = 'selector3(sequence2(hunger_over, selector3(sequence2(caught_prey, \'eat\'), sequence2(prey_nearby, ' \
                '\'go_to_prey\'), \'look_for_prey\')), sequence2(over_reproduction_age, \'reproduce\'), \'do_nothing\')'

pop_prey = None
pop_predator = None

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
    preys, predators = run_simulation(routine, predator_routine, lotka_volterra=True)
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


def show_behaviour(print_state: bool, lotka_voltera_model: bool, draw_grid: bool):
    prey_routine = gp.compile(best_prey, pset_prey)
    predator_routine = gp.compile(best_predator, pset_predator)
    preys, predators = run_simulation(prey_routine, predator_routine, print_state=print_state,
                                      lotka_volterra=lotka_voltera_model, draw_grid=draw_grid)
    return preys, predators


def decision_tree_prey():
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    toolbox_prey = create_toolbox(pset_prey, pool, eval_prey)
    pop_prey = toolbox_prey.population(n=50)
    hof_prey = tools.HallOfFame(1)
    stats_prey = create_stats()
    _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.7, 0.1, 10, stats_prey, halloffame=hof_prey)
    best_prey = hof_prey[0]
    nodes, edges, labels = gp.graph(hof_prey[0])
    save_tree("./graphs/prey_tree_grass2.txt", nodes, edges, labels)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    actions = PREY_TERMINALS
    G, _ = simplify_tree(0, G, edges, labels, actions)
    new_labels = {id: label for id, label in labels.items() if id in G.nodes()}
    plt.figure(figsize=(20, 10))
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, new_labels, font_size=12)
    plt.savefig("./graphs/prey_decision_tree2.png")
    plt.show()
    plot_tree(nodes, edges, labels)


def decision_tree_predator():
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    toolbox_predator = create_toolbox(pset_predator, pool, eval_predator)
    pop_predator = toolbox_predator.population(n=15)
    hof_predator = tools.HallOfFame(1)
    stats_predator = create_stats()
    _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.1, 5, stats_predator,
                                     halloffame=hof_predator)
    best_predator = hof_predator[0]
    nodes, edges, labels = gp.graph(hof_predator[0])
    save_tree("./graphs/pred_tree3.txt", nodes, edges, labels)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    actions = PREDATOR_TERMINALS
    G, _ = simplify_tree(0, G, edges, labels, actions)
    new_labels = {id: label for id, label in labels.items() if id in G.nodes()}
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, new_labels)
    plt.savefig("./graphs/predator_decision_tree3.png")
    plt.show()
    plot_tree(nodes, edges, labels)


if __name__ == '__main__':
    folder_path = "./map"
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    prey_counts, pred_counts = show_behaviour(print_state=True, lotka_voltera_model=True, draw_grid=False)

    # decision_tree_prey()
    # decision_tree_predator()

    np.savez(f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", prey=prey_counts, pred=pred_counts)
    plt.figure()
    plt.plot(range(len(prey_counts)), prey_counts, label='Preys')
    plt.plot(range(len(pred_counts)), pred_counts, label='Predators')
    plt.title("Population of preys and predators over time")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.yticks(range(0, max(prey_counts), 100))
    plt.legend()
    plt.savefig(f"results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
    plt.show()
