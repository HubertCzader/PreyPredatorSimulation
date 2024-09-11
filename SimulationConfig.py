class Config:
    preys = 300  # initial preys population
    predators = 100  # initial predators population

    # ------------------- Optimal parameters by equations ----------------------#
    # alpha = 0.6  # natural growth rate of preys
    # beta = 0.01  # death rate per encounter of preys due to predation
    # gamma = 0.001  # natural death rate of predators in the absence of food
    # delta = 0.2  # natural growth rate of predators
    #

    alpha = 0.9  # natural growth rate of preys
    beta = 0.01  # death rate per encounter of preys due to predation
    gamma = 0.001  # natural death rate of predators in the absence of food
    delta = 0.5  # natural growth rate of predators

    iterations = 1000  # number of iterations

