import numpy as np
import scipy as sp
from SimulationConfig import Config
from typing import List


def lotka_volterra(x, y, length):
    populations = [x, y]

    alpha = Config.alpha  # natural growth rate of preys
    beta = Config.beta  # death rate per encounter of preys due to predation
    gamma = Config.gamma  # natural death rate of predators in the absence of food
    delta = Config.delta  # natural growth rate of predators

    def rhs(populations, t) -> List:
        return [(alpha - beta * populations[1]) * populations[0], (gamma * populations[0] - delta) * populations[1]]

    t = np.arange(0, length)
    result = sp.integrate.odeint(rhs, populations, t)
    x, y = result[:, 0], result[:, 1]
    return x, y