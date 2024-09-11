import random as rnd


class Grass:
    ID = -1
    ptype = 0

    def __init__(self, x, y, reproductionRate):
        self.x = x
        self.y = y
        self.reproductionRate = reproductionRate

    def update(self):
        r = rnd.uniform(0, 1)
        offspring = 0
        if r < self.reproductionRate:
            offspring = Grass(self.x, self.y, self.reproductionRate)
        return offspring

