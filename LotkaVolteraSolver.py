import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List

# Initial values:
x0 = 300  # starting population of preys
y0 = 100  # starting population of predators
init = [x0, y0]

# alpha = 0.7  # natural growth rate of preys
# beta = 0.01  # death rate per encounter of preys due to predation
# gamma = 0.001  # natural death rate of predators in the absence of food
# delta = 0.1  # natural growth rate of predators

alpha = 0.7  # natural growth rate of preys
beta = 0.01  # death rate per encounter of preys due to predation
gamma = 0.001  # natural death rate of predators in the absence of food
delta = 0.2  # natural growth rate of predators


t = np.linspace(0, 100, 10000)


def rhs(populations, t) -> List:
    global alpha, beta, gamma, delta
    return [(alpha - beta * populations[1]) * populations[0], (gamma * populations[0] - delta) * populations[1]]


def generateModelPlot(result, time) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(time, result[:, 0], label="Preys")
    plt.plot(time, result[:, 1], label='Predators')
    plt.title("Lotka-Voltera Model")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    result = sp.integrate.odeint(rhs, init, t)
    generateModelPlot(result, t)
    # print(min(result[:, 1]))
