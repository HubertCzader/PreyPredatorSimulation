from GeneticProgramming import show_behaviour
from LotkaVolteraSolver import LotkaVolterra, generateModelPlot
from SimulationConfig import Config
from datetime import datetime
import numpy as np
import math


def mean_absolute_error(y_true, y_predicted):
    return np.mean(np.abs(y_true - y_predicted))


if __name__ == '__main__':
    results = LotkaVolterra(Config.preys, Config.predators, Config.iterations)
    correct_prey, correct_predator = np.array(results[:, 0], dtype=int), np.array(results[:, 1], dtype=int)
    min_error = math.inf
    best_results = None
    for i in range(100):
        print(f"Running simulation: {i + 1}")
        prey_counts, pred_counts = show_behaviour(print_state=False, lotka_voltera_model=True, draw_grid=False)
        if len(prey_counts) != Config.iterations + 1:
            continue
        prey_counts, pred_counts = np.array(prey_counts), np.array(pred_counts)
        mse = mean_absolute_error(prey_counts, correct_prey) + mean_absolute_error(pred_counts, correct_predator)
        if mse < min_error or best_results is None:
            min_error = mse
            best_results = [prey_counts, pred_counts]
    generateModelPlot(np.array(best_results).T, np.arange(0, Config.iterations + 1, 1))
    print("Mean Absolute Error: ", min_error)
    np.savez(f"results/bestresults{datetime.now().strftime('%Y-%m-%d_%H-%M')}", prey=best_results[0], pred=best_results[1])