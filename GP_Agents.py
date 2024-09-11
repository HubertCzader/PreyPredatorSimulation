import numpy as np
import random
from GrassAgent import Grass


def von_neumann_neighborhood(n):
    neighborhood = [(x, y) for x in range(-n, n + 1) for y in range(-n, n + 1) if abs(x) + abs(y) == n]
    random.shuffle(neighborhood)
    return neighborhood


def sigmoid(value, inflection_point, k=0.3):
    return 1 / (1 + np.exp(-k * (value - inflection_point)))


class Prey:
    ptype = -1  # 1 if predator, -1 for prey
    epsilon = 0.2

    def __init__(self, x_position, y_position, ID, lastAte, father, reproduction_age, death_age,
                 death_rate, reproduction_rate, weights, hunger_minimum, tree_function):

        self.x_position = x_position
        self.y_position = y_position
        self.ID = ID
        self.age = random.randint(0, death_age)
        self.lastAte = lastAte
        self.father = father
        self.reproduction_age = reproduction_age
        self.death_age = death_age
        self.death_rate = death_rate
        self.reproduction_rate = reproduction_rate
        self.weights = weights
        self.hunger_minimum = hunger_minimum
        self.q = 0
        self.tree_function = tree_function

    def in_grid(self, matrix, location):
        return -1 < location[0] < matrix.xDim and -1 < location[1] < matrix.yDim

    def predator_distance(self, matrix, location):
        for r in range(1, 4):
            for dx, dy in von_neumann_neighborhood(r):
                new_location = [location[0] + dx, location[1] + dy]
                if self.in_grid(matrix, new_location):
                    for entity in matrix.grid[new_location[0]][new_location[1]]:
                        if entity.ptype == 1:
                            return r
        return matrix.xDim

    def pick_action(self, matrix):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """
        grass_nearby = False
        grass_location = None
        own_location = np.array([self.x_position, self.y_position])
        known_predator_distance = self.predator_distance(matrix, own_location)
        location_predator_min_distance = known_predator_distance
        location_predator_max_distance = known_predator_distance
        furthest_from_predator_location = own_location
        on_grass = False
        for entity in matrix.grid[own_location[0]][own_location[1]]:
            if entity.ptype == 0:
                on_grass = True
                break
        for r in range(1, 3):
            for dx, dy in von_neumann_neighborhood(r):
                new_location = [self.x_position + dx, self.y_position + dy]
                if self.in_grid(matrix, new_location):
                    for entity in matrix.grid[new_location[0]][new_location[1]]:
                        if entity.ptype == 0 and grass_location is None:
                            grass_nearby = True
                            grass_location = new_location
                    if r < 3:
                        location_predator_distance = self.predator_distance(matrix, new_location)
                        if location_predator_distance > location_predator_max_distance:
                            location_predator_max_distance = location_predator_distance
                            furthest_from_predator_location = new_location
                        if location_predator_distance < location_predator_min_distance:
                            location_predator_min_distance = location_predator_distance

        result = self.tree_function(location_predator_min_distance < 4, grass_nearby,
                                    self.lastAte >= self.hunger_minimum,
                                    self.lastAte >= self.hunger_minimum and not grass_nearby,
                                    self.age >= self.reproduction_age, on_grass)

        if result == 'go_from_predator':
            return furthest_from_predator_location if furthest_from_predator_location is not None else own_location, -1, 0
        if result == 'go_to_food':
            return grass_location if grass_location is not None else own_location, -1, 0
        if result == "eat":
            return own_location, self.Eat(matrix.grid[self.x_position][self.y_position]), 0
        if result == "reproduce":
            return own_location, -1, self.Reproduce()
        if result == "look_for_food":
            possible_locations = []
            for r in range(1, 6):
                for dx, dy in von_neumann_neighborhood(r):
                    new_prey_location = [own_location[0] + dx, own_location[1] + dy]
                    if self.in_grid(matrix, new_prey_location):
                        possible_locations.append(new_prey_location)
            return random.choice(possible_locations), -1, 0
        return own_location, -1, 0

    def Aging(self, i):
        self.age += 1
        self.lastAte += 1
        self.epsilon = 1 / i
        return

    def Starve(self):
        if self.lastAte >= self.hunger_minimum:
            pdeath = self.lastAte * self.death_rate
        else:
            pdeath = self.death_rate
        r = np.random.rand()
        if r < pdeath + sigmoid(self.age, self.death_age, 0.8):
            return self.ID
        return -1

    def Eat(self, agentListAtMatrixPos):
        for agent in agentListAtMatrixPos:  # Not selected randomly at the moment, just eats the first prey in the list
            if type(agent) is Grass:
                self.lastAte = 0
                return agent.ID
        return -1

    def Reproduce(self):
        offspring = 0
        r = np.random.rand()
        if self.age >= self.reproduction_age and r < self.reproduction_rate and self.lastAte < self.hunger_minimum:
            food_in_stomach = self.hunger_minimum - self.lastAte
            offspring_food = food_in_stomach // 2
            self.lastAte = self.hunger_minimum - food_in_stomach + offspring_food
            offspring_last_ate = self.hunger_minimum - offspring_food if self.hunger_minimum - offspring_food < self. \
                hunger_minimum - 1 else self.hunger_minimum - 1
            offspring = Prey(self.x_position, self.y_position, -1, offspring_last_ate, self.ID,
                             self.reproduction_age, self.death_age,
                             self.death_rate, self.reproduction_rate, self.weights,
                             self.hunger_minimum, self.tree_function)  # ID is changed in Grid.update()
            offspring.age = 0
        return offspring


class Predator:
    ptype = 1  # 1 if predator, -1 for prey
    epsilon = 0.2

    def __init__(self, x_position, y_position, ID, lastAte, father, reproduction_age, death_age,
                 death_rate, reproduction_rate, weights, hunger_minimum, tree_function):

        self.x_position = x_position
        self.y_position = y_position
        self.ID = ID
        self.age = random.randint(0, death_age)
        self.lastAte = lastAte  # Time when predator last ate
        self.father = father
        self.reproduction_age = reproduction_age
        self.death_age = death_age
        self.death_rate = death_rate
        self.reproduction_rate = reproduction_rate
        self.weights = weights
        self.hunger_minimum = hunger_minimum
        self.q = 0
        self.tree_function = tree_function

    def in_grid(self, matrix, location):
        return -1 < location[0] < matrix.xDim and -1 < location[1] < matrix.yDim

    def pick_action(self, matrix):
        """
        Perform action (i.e. movement) of the agent depending on its evaluations
        """

        prey_nearby = False
        prey_location = None
        own_location = np.array([self.x_position, self.y_position])
        for entity in matrix.grid[own_location[0]][own_location[1]]:
            if entity.ptype == -1:
                prey_nearby = True
                prey_location = own_location.copy()
                break
        if prey_location is None:
            for r in range(1, 6):
                for dx, dy in von_neumann_neighborhood(r):
                    new_location = [own_location[0] + dx, own_location[1] + dy]
                    if self.in_grid(matrix, new_location):
                        for entity in matrix.grid[new_location[0]][new_location[1]]:
                            if entity.ptype == -1:
                                if r < 4:
                                    prey_nearby = True
                                prey_location = new_location.copy()
                                break
                if prey_location:
                    break

        if self.tree_function is None:
            return own_location, -1, 0

        result = self.tree_function(prey_nearby,
                                    self.lastAte >= self.hunger_minimum,
                                    self.lastAte >= self.hunger_minimum and not prey_nearby,
                                    self.age >= self.reproduction_age,
                                    prey_location is not None and prey_location[0] == own_location[0] and prey_location[
                                        1] == own_location[1],
                                    )

        if result == 'go_to_prey':
            if prey_location is None:
                return own_location, -1, 0
            if prey_location[0] == own_location[0] and prey_location[1] == own_location[1]:
                result = "eat"
            else:
                x = prey_location[0] - own_location[0]
                x = x if x == 0 else x // abs(x)
                y = prey_location[1] - own_location[1]
                y = y if y == 0 else y // abs(y)
                return own_location + np.array([x, y]), -1, 0

                # x = prey_location[0] - own_location[0]
                # x = x if x == 0 else x // abs(x) * min(abs(x), 2)
                # y = prey_location[1] - own_location[1]
                # y = y if y == 0 else y // abs(y) * min(abs(y), 2)
                # return own_location + np.array([x, y]), -1, 0
        if result == "eat":
            return own_location, self.Eat(matrix.grid[self.x_position][self.y_position]), 0
        if result == "look_for_prey":
            possible_locations = []
            for r in range(1, 6):
                for dx, dy in von_neumann_neighborhood(r):
                    new_predator_location = [own_location[0] + dx, own_location[1] + dy]
                    if self.in_grid(matrix, new_predator_location):
                        possible_locations.append(new_predator_location)
            return random.choice(possible_locations), -1, 0
        if result == "reproduce":
            return own_location, -1, self.Reproduce()
        return own_location, -1, 0

    def Aging(self, i):
        self.age += 1
        self.lastAte += 1
        return

    def Eat(self, agentListAtMatrixPos):
        for agent in agentListAtMatrixPos:
            if type(agent) is Prey:  # Not selected randomly at the moment, just eats the first prey in the list
                r = random.random()
                hunger_influence = sigmoid(self.lastAte, self.hunger_minimum, 2)
                success_rate = hunger_influence * (1 - np.exp(-self.death_rate * self.lastAte * 12))
                if r < success_rate:
                    self.lastAte = 0
                    return agent.ID
        return -1

    def Starve(self):
        if self.lastAte >= self.hunger_minimum:
            pdeath = self.lastAte * self.death_rate
        else:
            pdeath = self.death_rate
        r = np.random.rand()
        if r < pdeath + sigmoid(self.age, self.death_age, 0.8):
            return self.ID
        return -1

    def Reproduce(self):
        offspring = 0
        r = np.random.rand()
        # This peace of code makes reproduction dependent on the level of satiety
        # if self.age >= self.reproduction_age and self.lastAte < (self.hunger_minimum // 2):
        if self.age >= self.reproduction_age and r < self.reproduction_rate and self.lastAte < self.hunger_minimum:
            food_in_stomach = self.hunger_minimum - self.lastAte
            offspring_food = food_in_stomach // 2

            self.lastAte = self.hunger_minimum - food_in_stomach + offspring_food
            offspring_last_ate = self.hunger_minimum - offspring_food if self.hunger_minimum - offspring_food < self. \
                hunger_minimum - 1 else self.hunger_minimum - 1
            offspring = Predator(self.x_position, self.y_position, -1, offspring_last_ate, self.ID,
                                 self.reproduction_age, self.death_age,
                                 self.death_rate, self.reproduction_rate, self.weights,
                                 self.hunger_minimum, self.tree_function)  # ID is changed in Grid.update()

            # offspring = Predator(self.x_position, self.y_position, -1, 0, self.ID,
            #                      self.reproduction_age, self.death_age,
            #                      self.death_rate, self.reproduction_rate, self.weights,
            #                      self.hunger_minimum, self.tree_function)

            offspring.age = 0
        return offspring
