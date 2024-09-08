import matplotlib.pyplot as plt
import numpy as np

def plot_from_file(filename):
    plt.figure()
    data = np.load(filename)
    preys = data['prey']
    predators = data['pred']
    plt.plot(range(len(preys)), preys, label='Preys')
    plt.plot(range(len(predators)), predators, label='Predators', color="red")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.yticks(range(0, max(preys), 100))
    plt.title("Population of preys and predators over time")
    plt.legend()
    plt.show(block=False)


if __name__ == "__main__":
    path = "results/400_3_5.npz"
    plot_from_file(path)


