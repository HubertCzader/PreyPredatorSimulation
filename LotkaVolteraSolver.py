import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from typing import List

# Initial values:
x0 = 300  # starting population of preys
y0 = 100  # starting population of predators
init = [x0, y0]

alpha = 0.7  # natural growth rate of preys
beta = 0.01  # death rate per encounter of preys due to predation
gamma = 0.001  # natural death rate of predators in the absence of food
delta = 0.5  # natural growth rate of predators

t = np.linspace(0, 1001, 1001)

def rhs(populations, t) -> List:
    global alpha, beta, gamma, delta
    return [(alpha - beta * populations[1]) * populations[0], (gamma * populations[0] - delta) * populations[1]]


def LotkaVolterra(x, y, length):
    populations = [x, y]
    t = np.linspace(0, length, 1001)
    return sp.integrate.odeint(rhs, populations, t)


def generateModelPlot(result, time) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(time, result[:, 0], label="Preys")
    plt.plot(time, result[:, 1], label='Predators', color='red')
    # plt.title("Lotka-Voltera Model")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend(loc="upper right")

    parameter_text = f"α = {alpha}\nβ = {beta}\nγ = {gamma}\nδ = {delta}"
    anchored_text = AnchoredText(parameter_text, loc='upper left', frameon=True, pad=0.5, borderpad=1)
    anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.5")
    plt.gca().add_artist(anchored_text)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    result = sp.integrate.odeint(rhs, init, t)
    print(result[:, 0])
    # generateModelPlot(result, t)
