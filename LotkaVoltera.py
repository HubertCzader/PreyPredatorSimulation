import matplotlib.pyplot as plt
import numpy as np


def generateModelPlot(preys, predators) -> None:
    plt.figure()
    plt.plot(np.arange(1, len(preys) + 1), preys, label="Preys")
    plt.plot(np.arange(1, len(predators) + 1), predators, label='Predators')
    plt.title("Lotka-Voltera Model")
    plt.xlabel("Epochs")
    plt.ylabel("Population")
    plt.legend(loc="upper right")
    plt.show()


def lotka_volterra(x, y, length):
    xs = [x]  # preys
    ys = [y]  # predators

    alpha = 0.03  # natural growth rate of preys
    beta = 0.001  # death rate per encounter of preys due to predation
    gamma = 0.001  # natural growth rate of predators
    delta = 0.05  # natural death rate of predators in the absence of food

    timestep = 0.01

    steps = int(length / timestep)
    inverse_timestep = int(1 / timestep)

    for i in range(steps):
        dx = alpha * x - beta * x * y
        dy = gamma * x * y - delta * y

        # Calculating next element by Euler's method
        x += dx * timestep
        y += dy * timestep

        if (i + 1) % inverse_timestep == 0 and len(xs) < length:
            xs.append(int(x))
            ys.append(int(y))

    generateModelPlot(xs, ys)
    return xs, ys
