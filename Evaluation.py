import numpy as np
from SimulationConfig import Config
from LotkaVolteraSolver import LotkaVolterra, generateModelPlot


def load_data(filename):
    data = np.load(filename)
    preys = data['prey']
    predators = data['pred']
    return preys, predators


def mean_absolute_error(y_true, y_predicted):
    return np.mean(np.abs(y_true - y_predicted))


def mean_squared_error(y_true, y_predicted):
    return np.mean(np.square(y_true - y_predicted))


if __name__ == "__main__":
    preys, predators = load_data("results/bestresults2021-06-01_20-00.npz")
    results = LotkaVolterra(Config.preys, Config.predators, Config.iterations)
    correct_prey, correct_predator = np.array(results[:, 0], dtype=int), np.array(results[:, 1], dtype=int)
    mse = mean_squared_error(preys, correct_prey) + mean_squared_error(predators, correct_predator)
    mae = mean_absolute_error(preys, correct_prey) + mean_absolute_error(predators, correct_predator)
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)
    generateModelPlot(results, np.arange(0, Config.iterations + 1, 1))

