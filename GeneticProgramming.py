import multiprocessing
import networkx as nx
import numpy as np

from deap import gp, creator, base, tools, algorithms
from matplotlib import pyplot as plt
from datetime import datetime

from Simulation import run_simulation, fitness_function
from LotkaVoltera import lotka_volterra
from TreeUtils import simplify_tree
from Utils import create_pset, create_pset_pred, create_stats, create_toolbox, plot_logbook, plot_tree, plot_logbook_box
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

best_prey = 'selector4(sequence2(predator_nearby, \'go_from_predator\'), sequence2(hunger_over, ' \
            'selector3(sequence2(found_food, \'eat\'), sequence2(food_nearby, \'go_to_food\'), \'look_for_food\')), ' \
            'sequence2(over_reproduction_age, \'reproduce\'), \'do_nothing\')'

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
    _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.7, 0.1, 10, stats_prey, halloffame=hof_prey)
    nodes, edges, labels = gp.graph(hof_prey[0])
    save_tree("./graphs/prey_tree2.txt", nodes, edges, labels)
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


def plot_fitness_prey(population):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    toolbox_prey = create_toolbox(pset_prey, pool, eval_prey)
    pop_prey = toolbox_prey.population(n=population)
    hof_prey = tools.HallOfFame(1)
    stats_prey = create_stats()
    _, logbook = algorithms.eaSimple(pop_prey, toolbox_prey, 0.5, 0.2, 20, stats_prey, halloffame=hof_prey)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fig = plot_logbook(logbook)
    plt.show()
    fig.savefig(f"./gp_results/grass_prey_fitness_{population}_{current_time}.png")
    box = plot_logbook_box(logbook)
    plt.show()
    box.savefig(f"./gp_results/grass_prey_box_{population}_{current_time}.png")
    nodes, edges, labels = gp.graph(hof_prey[0])
    save_tree(f"./gp_results/grass_prey_tree_{population}_{current_time}.txt", nodes, edges, labels)
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
    plt.savefig(f"./gp_results/grass_prey_tree_{population}_{current_time}.png")
    plt.show()
    plot_tree(nodes, edges, labels)
    pool.close()
    pool.join()

def plot_fitness_predator(population):
    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    toolbox_predator = create_toolbox(pset_predator, pool, eval_predator)
    pop_predator = toolbox_predator.population(n=population)
    hof_predator = tools.HallOfFame(1)
    stats_predator = create_stats()
    _, logbook = algorithms.eaSimple(pop_predator, toolbox_predator, 0.5, 0.2, 20, stats_predator,
                                     halloffame=hof_predator)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fig = plot_logbook(logbook)
    plt.show()
    fig.savefig(f"./gp_results/grass_predator_fitness_{population}_{current_time}.png")
    box = plot_logbook_box(logbook)
    plt.show()
    box.savefig(f"./gp_results/grass_predator_box_{population}_{current_time}.png")
    nodes, edges, labels = gp.graph(hof_predator[0])
    save_tree(f"./gp_results/grass_predator_tree_{population}_{current_time}.txt", nodes, edges, labels)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    actions = PREDATOR_TERMINALS
    G, _ = simplify_tree(0, G, edges, labels, actions)
    new_labels = {id: label for id, label in labels.items() if id in G.nodes()}
    plt.figure(figsize=(20, 10))
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, new_labels, font_size=12)
    plt.savefig(f"./gp_results/predator_tree_{population}_{current_time}.png")
    plt.show()
    plot_tree(nodes, edges, labels)
    pool.close()
    pool.join()


if __name__ == "__main__":
    for i in range(5):
        print(f"Run {i+1}\n")
        for n in [10, 20, 50]:
            plot_fitness_prey(n)

        for n in [10, 20, 50]:
            plot_fitness_predator(n)