import numpy as np
import argparse
import random

from matplotlib import pyplot as plt

from GP_Agents import Prey, Predator
from SimulationConfig import Config
from GrassAgent import Grass


def fitness_function(prey_function):
    return np.mean([run_simulation(prey_function) for _ in range(3)])


def run_simulation(prey_function, pred_function, print_state=False, draw_grid=False, lotka_volterra=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gridDim', default=50, type=int, help='Size of the grid')
    parser.add_argument('--nPredators', default=Config.predators, type=int, help='Number of initial predators')
    parser.add_argument('--nPrey', default=Config.preys, type=int, help='Number of initial preys')
    parser.add_argument('--nGrass', default=1000, type=int, help='Number of initial grass')
    parser.add_argument('--predRepAge', default=5, type=int, help='Reproduction Age of predators')
    parser.add_argument('--predDeathAge', default=30, type=int, help='Death Age of predators')
    parser.add_argument('--predDeathRate', default=Config.gamma, type=int, help='Probability of dying by hunger')
    parser.add_argument('--predRepRate', default=Config.delta, type=int, help='Probability of giving birth by predators')
    parser.add_argument('--preyRepAge', default=3, type=int, help='Reproduction Age of preys')
    parser.add_argument('--preyDeathAge', default=30, type=int, help='Death Age of preys')
    parser.add_argument('--preyDeathRate', default=Config.beta, type=int, help='Probability of dying due to predation')
    parser.add_argument('--preyRepRate', default=Config.alpha, type=int, help='Probability of giving birth by preys')
    parser.add_argument('--mPred', default=3, type=int, help='The time after which predators get hungry')
    parser.add_argument('--mPrey', default=4, type=int, help='The time after which prey get hungry')
    parser.add_argument('--grassRepRate', default=0.25, type=int, help='Probability of giving birth')
    parser.add_argument('--totalNumIterations', default=Config.iterations, type=int)

    args = parser.parse_args()

    xDim = args.gridDim
    yDim = args.gridDim
    nPredators = args.nPredators
    nPrey = args.nPrey
    nGrass = args.nGrass
    predRepAge = args.predRepAge
    predDeathAge = args.predDeathAge
    predDeathRate = args.predDeathRate
    predRepRate = args.predRepRate
    preyRepAge = args.preyRepAge
    preyDeathAge = args.preyDeathAge
    preyDeathRate = args.preyDeathRate
    preyRepRate = args.preyRepRate
    mPred = args.mPred
    mPrey = args.mPrey
    grassRepRate = args.grassRepRate

    totalNumIterations = args.totalNumIterations
    all_epochs_num_agents = []

    preyV = [nPrey]
    predV = [nPredators]
    grassV = [nGrass]
    predLastAteV = []
    preyLastAteV = []
    ratioV = []

    grid = Grid(xDim, yDim, nPredators, nPrey, nGrass, predRepAge, predDeathAge, predDeathRate, predRepRate, preyRepAge,
                preyDeathAge, preyDeathRate, preyRepRate, mPred, mPrey, grassRepRate, prey_function, pred_function)

    if print_state:
        print("Iteration: %d. Preys: %d, Predators: %d, Grass: %d " % (0, nPrey, nPredators, nGrass))

    for i in range(1, totalNumIterations + 1):
        if draw_grid:
            grid.draw()
        numAgents = grid.update(i)
        grid.save_img(i)
        if draw_grid:
            grid.draw()
        if numAgents[0] == 0 or numAgents[1] == 0:
            break
        preyV.append(numAgents[0])
        predV.append(numAgents[1])
        grassV.append(numAgents[2])

        [preyDeathAvg, predDeathAvg, preyLastAteP, predLastAteP, ratio] = numAgents[3:]
        if print_state:
            print("Iteration: %d. Preys: %d, Predators: %d, Grass: %d " % (i, numAgents[0], numAgents[1], numAgents[2]))
        preyLastAteV.append(preyLastAteP)
        predLastAteV.append(predLastAteP)
        ratioV.append(ratio)
        all_epochs_num_agents.append(numAgents.copy())

    if lotka_volterra:
        return preyV, predV

    fitness = lambda values: sum([((i + 1) ** 2) * value / 400 for _, value in enumerate(values[-20:])])
    return fitness(preyV), fitness(predV)


class Grid:
    def __init__(self, xDim, yDim, nPredators, nPrey, nGrass, predRepAge, predDeathAge, predDeathRate, predRepRate,
                 preyRepAge,
                 preyDeathAge, preyDeathRate, preyRepRate, mPred, mPrey, grassRepRate, prey_function,
                 pred_function):
        self.xDim = xDim
        self.yDim = yDim
        self.nPredators = nPredators
        self.nPrey = nPrey
        self.grid = [[[] for x in range(xDim)] for y in range(yDim)]
        self.grassGrid = [[0 for x in range(xDim)] for y in range(yDim)]
        self.agentList = []
        self.ID = 1
        self.preyDeaths = 0
        self.predDeaths = 0
        self.preyDeathAgeSum = 0
        self.predDeathAgeSum = 0
        self.numPred = nPredators
        self.numPrey = nPrey
        self.numGrass = nGrass
        self.numPrey = nPrey
        self.numGrass = nGrass
        self.grassRepRate = grassRepRate

        self.grassLimit = 1000
        self.fieldsWithoutGrass = [(x, y) for x in range(xDim) for y in range(yDim)]

        for i in range(nPredators):
            initWeights = np.random.rand(12) * 6 - 3
            x = random.randint(0, xDim - 1)
            y = random.randint(0, yDim - 1)
            lastAte = random.randint(0, mPred + 2)
            pred = Predator(x, y, self.ID, lastAte, 0, predRepAge, predDeathAge, predDeathRate,
                            predRepRate, initWeights, mPred, pred_function)
            self.grid[x][y].append(pred)
            self.agentList.append([self.ID, x, y, 0])
            self.ID += 1
        for i in range(nPrey):
            initWeights = np.random.rand(12) * 6 - 3
            x = random.randint(0, xDim - 1)
            y = random.randint(0, yDim - 1)
            lastAte = random.randint(0, mPrey + 2)
            prey = Prey(x, y, self.ID, lastAte, 0, preyRepAge, preyDeathAge, preyDeathRate,
                        preyRepRate, initWeights, mPrey, prey_function)
            self.grid[x][y].append(prey)
            self.agentList.append([self.ID, x, y, 1])
            self.ID += 1
        for i in range(nGrass):
            x, y = random.choice(self.fieldsWithoutGrass)
            self.fieldsWithoutGrass.remove((x, y))
            # x = random.randint(0, self.xDim - 1)
            # y = random.randint(0, self.yDim - 1)
            grass = Grass(x, y, self.grassRepRate)
            grass.ID = self.ID
            self.grid[x][y].append(grass)
            self.grassGrid[x][y] += 1
            self.agentList.append([self.ID, x, y, 2])
            self.ID += 1

    def update(self, i):
        if self.numGrass == 0:
            for i in range(50):
                x, y = random.choice(self.fieldsWithoutGrass)
                self.fieldsWithoutGrass.remove((x, y))
                # x = random.randint(0, self.xDim - 1)
                # y = random.randint(0, self.yDim - 1)
                grass = Grass(x, y, self.grassRepRate)
                grass.ID = self.ID
                self.grid[x][y].append(grass)
                self.numGrass += 1
                self.grassGrid[x][y] += 1
                self.agentList.append([self.ID, x, y, 2])
                self.ID += 1

        random.shuffle(self.agentList)
        predLastAte = 0
        preyLastAte = 0
        deathsbyeat = 0
        deathsbystv = 0
        for agentInfo in self.agentList:
            agentId = agentInfo[0]
            x = agentInfo[1]
            y = agentInfo[2]
            agentType = agentInfo[3]
            agent = None
            # Get current agent based on id
            for agents in self.grid[x][y]:
                if agents.ID == agentId:
                    agent = agents
                    break
            # Move agents, add eating etc.
            if agentType == 0:  # Predator
                predLastAte += agent.lastAte
                agent.Aging(i)
                # Moving and learning
                [newCoordsX, newCoordsY], eatenID, offspring = agent.pick_action(self)
                newCoordsX = int(newCoordsX)
                newCoordsY = int(newCoordsY)
                self.grid[x][y].remove(agent)
                agent.x_position = newCoordsX
                agent.y_position = newCoordsY
                # print(newCoordsX, newCoordsY)
                self.grid[newCoordsX][newCoordsY].append(agent)
                agentInfo[1] = newCoordsX
                agentInfo[2] = newCoordsY
                x = newCoordsX
                y = newCoordsY
                eatenAgent = None
                if eatenID != -1:
                    for agents in self.grid[x][y]:
                        if agents.ID == eatenID:
                            eatenAgent = agents
                            break
                    self.preyDeathAgeSum += eatenAgent.age
                    self.preyDeaths += 1
                    self.numPrey -= 1
                    self.grid[x][y].remove(eatenAgent)
                    deathsbyeat = deathsbyeat + 1
                    for agentProperties in self.agentList:
                        if agentProperties[0] == eatenID:
                            eatenAgent = agentProperties
                            break
                    self.agentList.remove(eatenAgent)
                else:
                    if agent.Starve() != -1:
                        self.predDeathAgeSum += agent.age
                        self.predDeaths += 1
                        self.numPred -= 1
                        self.grid[x][y].remove(agent)
                        self.agentList.remove(agentInfo)

                if offspring != 0:
                    offspring.ID = self.ID
                    offspring.age = 0
                    offspring.epsilon = agent.epsilon
                    self.numPred += 1
                    self.grid[x][y].append(offspring)
                    self.agentList.append([self.ID, x, y, 0])
                    self.ID += 1

            elif agentType == 1:  # Prey
                preyLastAte += agent.lastAte
                agent.Aging(i)
                # Monving and learning
                [newCoordsX, newCoordsY], eatenID, offspring = agent.pick_action(self)
                newCoordsX = int(newCoordsX)
                newCoordsY = int(newCoordsY)
                self.grid[x][y].remove(agent)
                agent.x_position = newCoordsX
                agent.y_position = newCoordsY
                self.grid[newCoordsX][newCoordsY].append(agent)
                agentInfo[1] = newCoordsX
                agentInfo[2] = newCoordsY
                x = newCoordsX
                y = newCoordsY

                if eatenID != -1:
                    eatenAgent = None
                    for agents in self.grid[x][y]:
                        if agents.ID == eatenID:
                            eatenAgent = agents
                            break
                    self.grid[x][y].remove(eatenAgent)
                    self.grassGrid[x][y] -= 1
                    if self.grassGrid[x][y] == 0:
                        self.fieldsWithoutGrass.append((x, y))
                    self.numGrass -= 1
                    for agentProperties in self.agentList:
                        if agentProperties[0] == eatenID:
                            eatenAgent = agentProperties
                            break
                    self.agentList.remove(eatenAgent)
                else:
                    if agent.Starve() != -1:
                        self.preyDeathAgeSum += agent.age
                        self.preyDeaths += 1
                        self.numPrey -= 1
                        self.grid[x][y].remove(agent)
                        self.agentList.remove(agentInfo)
                        deathsbystv = deathsbystv + 1

                if offspring != 0:
                    offspring.ID = self.ID
                    offspring.age = 0
                    offspring.epsilon = agent.epsilon
                    self.numPrey += 1
                    self.grid[x][y].append(offspring)
                    self.agentList.append([self.ID, x, y, 1])
                    self.ID += 1

            elif agentType == 2:  # Grass
                if self.numGrass < self.grassLimit:
                    offspring = agent.update()
                    if offspring != 0:
                        offspring.ID = self.ID
                        if len(self.fieldsWithoutGrass) != 0:
                            offspring.x, offspring.y = random.choice(self.fieldsWithoutGrass)
                            self.fieldsWithoutGrass.remove((offspring.x, offspring.y))
                        else:
                            offspring.x = random.randint(0, self.xDim - 1)
                            offspring.y = random.randint(0, self.yDim - 1)
                        self.grassGrid[offspring.x][offspring.y] += 1
                        self.numGrass += 1
                        self.grid[offspring.x][offspring.y].append(offspring)
                        self.agentList.append([self.ID, offspring.x, offspring.y, 2])
                        self.ID += 1

        if self.preyDeaths != 0:
            preyDeathAvg = self.preyDeathAgeSum / self.preyDeaths
        else:
            preyDeathAvg = 0
        if self.predDeaths != 0:
            predDeathAvg = self.predDeathAgeSum / self.predDeaths
        else:
            predDeathAvg = 0
        if self.numPrey != 0:
            preyLastAteP = float(preyLastAte) / float(self.numPrey)
        else:
            preyLastAteP = 0
        if self.numPred != 0:
            predLastAteP = float(predLastAte) / float(self.numPred)
        else:
            predLastAteP = 0
        ratio = deathsbyeat / (deathsbystv + deathsbyeat + 0.00000000001)
        return [self.numPrey, self.numPred, self.numGrass, preyDeathAvg, predDeathAvg, preyLastAteP, predLastAteP,
                ratio]

    def save_img(self, i):
        xs = [[], [], []]
        ys = [[], [], []]

        conflict_xs = []
        conflict_ys = []

        for agent in self.agentList:
            x = agent[1]
            y = agent[2]
            agent_type = agent[3]

            if agent_type == 0:
                xs[0].append(x)
                ys[0].append(y)
            elif agent_type == 1:
                xs[1].append(x)
                ys[1].append(y)
            else:
                xs[2].append(x)
                ys[2].append(y)

        for x_pred, y_pred in zip(xs[0], ys[0]):
            if (x_pred, y_pred) in zip(xs[1], ys[1]):
                conflict_xs.append(x_pred)
                conflict_ys.append(y_pred)

        for gx, gy in zip(conflict_xs, conflict_ys):
            if gx in xs[0] and gy in ys[0]:
                idx = xs[0].index(gx)
                xs[0].pop(idx)
                ys[0].pop(idx)
            if gx in xs[1] and gy in ys[1]:
                idx = xs[1].index(gx)
                xs[1].pop(idx)
                ys[1].pop(idx)

        plt.clf()
        plt.scatter(xs[2], ys[2], color='g', label="Grass")
        plt.scatter(xs[1], ys[1], color='b', label="Preys")
        plt.scatter(xs[0], ys[0], color='r', label="Predators")
        plt.scatter(conflict_xs, conflict_ys, color='orange', label="Conflict")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)
        plt.savefig("map/epoch" + str(i) + ".png")

    def draw(self):
        # plt.clf()
        xs = [[], [], []]
        ys = [[], [], []]
        for agents in self.agentList:
            x = agents[1]
            y = agents[2]
            type = agents[3]
            if type == 0:
                xs[0].append(x)
                ys[0].append(y)
            elif type == 1:
                xs[1].append(x)
                ys[1].append(y)
            else:
                xs[2].append(x)
                ys[2].append(y)

        plt.clf()
        plt.scatter(xs[2], ys[2], color='g', label="Grass")
        plt.scatter(xs[1], ys[1], color='b', label="Preys")
        plt.scatter(xs[0], ys[0], color='r', label="Predators")
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)
        plt.axis([-1, self.xDim, -1, self.yDim])
        plt.pause(0.1)
        plt.draw()
        # plt.show()


