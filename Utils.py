import numpy as np
from deap import gp, creator, base, tools
import networkx as nx
import matplotlib.pyplot as plt
from FlowOperators import sequence2, sequence3, selector2, selector3, selector4, randomSelector2
from adjustText import adjust_text

# def plot_logbook(logbook):
#     min_values = logbook.select("min")
#     max_values = logbook.select("max")
#     avg_values = logbook.select("avg")
#     std_values = logbook.select("std")
#     epoch_values = np.arange(len(avg_values))
#     plt.errorbar(epoch_values, avg_values, std_values, label="avg +- std", ls='none', capsize=3, fmt='o')
#     plt.plot(epoch_values, min_values, "-o", label="min")
#     plt.plot(epoch_values, max_values, "-o", label="max")
#     plt.legend()
#     plt.show()

def plot_logbook(logbook):
    min_values = logbook.select("min")
    max_values = logbook.select("max")
    avg_values = logbook.select("avg")
    std_values = logbook.select("std")
    epoch_values = np.arange(len(avg_values))
    fig, ax = plt.subplots()
    ax.errorbar(epoch_values, avg_values, std_values, label="avg +- std", ls='none', capsize=3, fmt='o')
    ax.plot(epoch_values, min_values, "-o", label="min")
    ax.plot(epoch_values, max_values, "-o", label="max")
    ax.legend()
    return fig


def plot_logbook_box(logbook):
    # Wybieramy odpowiednie wartości
    min_values = logbook.select("min")
    max_values = logbook.select("max")
    avg_values = logbook.select("avg")
    std_values = logbook.select("std")
    epoch_values = np.arange(len(avg_values))
    lower_box = np.array(avg_values) - np.array(std_values)
    upper_box = np.array(avg_values) + np.array(std_values)

    fig, ax = plt.subplots()

    ax.errorbar(epoch_values, avg_values,
                yerr=[np.array(avg_values) - np.array(min_values), np.array(max_values) - np.array(avg_values)],
                fmt='o',
                color='black', ecolor='black', capsize=5, label='min to max', zorder=1)
    ax.vlines(epoch_values, lower_box, upper_box, color='dodgerblue', lw=8, alpha=0.6, label='avg ± std', zorder=2)
    ax.plot(epoch_values, avg_values, 'o', color='#1C39BB', label='avg', zorder=3)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness value')
    ax.legend(loc='lower right')

    return fig


def plot_tree(nodes, edges, labels):
    plt.figure(figsize=(25, 15))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)

    # texts = []
    # for node, label in labels.items():
    #     x, y = pos[node]
    #     texts.append(plt.text(x, y, label, ha='center', va='center'))
    #
    # adjust_text(texts, force_points=0.5, force_text=0.5, expand_points=(1.2, 1.5), expand_text=(1.2, 1.5),
    #             arrowprops=dict(arrowstyle='->', color='red'))

    nx.draw_networkx_labels(g, pos, labels, font_size=8)
    plt.show()


def create_pset(terminals, args):
    pset = gp.PrimitiveSet("main", len(args))
    pset.addPrimitive(sequence2, 2)
    pset.addPrimitive(sequence3, 3)
    pset.addPrimitive(selector2, 2)
    pset.addPrimitive(selector3, 3)
    pset.addPrimitive(selector4, 4)
    deap_args = [f"ARG{i}" for i in range(len(args))]
    kargs = {k: v for k, v in zip(deap_args, args)}
    pset.renameArguments(**kargs)
    for terminal in terminals:
        pset.addTerminal(terminal)
    return pset


def create_pset_pred(terminals, args):
    pset = gp.PrimitiveSet("main", len(args))
    pset.addPrimitive(sequence2, 2)
    pset.addPrimitive(sequence3, 3)
    pset.addPrimitive(selector2, 2)
    pset.addPrimitive(selector3, 3)
    deap_args = [f"ARG{i}" for i in range(len(args))]
    kargs = {k: v for k, v in zip(deap_args, args)}
    pset.renameArguments(**kargs)
    for terminal in terminals:
        pset.addTerminal(terminal)
    return pset


def create_toolbox(pset, pool, eval_fn):
    toolbox = base.Toolbox()
    toolbox.register("map", pool.map)

    toolbox.register(f"expr_init", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register(f"individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register(f"population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    return toolbox


def create_stats():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats
