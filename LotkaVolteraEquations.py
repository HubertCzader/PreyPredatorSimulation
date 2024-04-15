import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import List

# Initial values:
x0 = 120  # starting population of preys
y0 = 100  # starting population of predators
init = [x0, y0]
a = 0.03  # natural growth rate of preys
b = 0.001  # death rate per encounter of preys due to predation
c = 0.001  # natural growth rate of predators
d = 0.05  # natural death rate of predators in the absence of food
t = np.linspace(0, 1000, 2500)



def rhs(populations, t) -> List:
    global a, b, c, d
    return [(a - b * populations[1]) * populations[0], (c * populations[0] - d) * populations[1]]


def generateModelPlot(result, time) -> None:
    plt.figure()
    plt.plot(time, result[:, 0], label="Preys")
    plt.plot(time, result[:, 1], label='Predators')
    plt.title("Lotka-Voltera Model")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    result = sp.integrate.odeint(rhs, init, t)
    generateModelPlot(result, t)
    print(min(result[:, 1]))
