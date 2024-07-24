import multiprocessing
import networkx as nx
import numpy as np

from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt

from Simulation import run_simulation
from LotkaVoltera import lotka_volterra
from TreeUtils import simplify_tree
from Utils import create_pset, create_stats, create_toolbox, plot_logbook, plot_tree


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

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
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
    pop_prey = toolbox_prey.population(n=10)
    hof_prey = tools.HallOfFame(1)
    stats_prey = create_stats()
    _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.5, 0.2, 5, stats_prey, halloffame=hof_prey)
    best_prey = hof_prey[0]
    nodes, edges, labels = gp.graph(hof_prey[0])
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    actions = PREY_TERMINALS
    G, _ = simplify_tree(0, G, edges, labels, actions)
    new_labels = {id: label for id, label in labels.items() if id in G.nodes()}
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, new_labels)
    plt.savefig("prey_decision_tree.png")
    # plt.show()
    # plot_tree(nodes, edges, labels)


def decision_tree_predator():
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    toolbox_predator = create_toolbox(pset_predator, pool, eval_predator)
    pop_predator = toolbox_predator.population(n=50)
    hof_predator = tools.HallOfFame(1)
    stats_predator = create_stats()
    _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.3, 10, stats_predator, halloffame=hof_predator)
    best_predator = hof_predator[0]
    nodes, edges, labels = gp.graph(hof_predator[0])

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
    plt.savefig("predator_decision_tree.png")
    # plt.show()
    # plot_tree(nodes, edges, labels)


if __name__ == '__main__':
    prey_counts, pred_counts = show_behaviour(print_state=True, lotka_voltera_model=True, draw_grid=False)
    # decision_tree_prey()
    # decision_tree_predator()
    # correct_prey, correct_pred = lotka_volterra(prey_counts[0], pred_counts[0], len(prey_counts))
    plt.figure()
    plt.plot(range(len(prey_counts)), prey_counts, label='Preys')
    plt.plot(range(len(pred_counts)), pred_counts, label='Predators')
    plt.title("Population of preys and predators over time")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
