class Config:
    preys = 300  # initial preys population
    predators = 100  # initial predators population

    alpha = 0.027  # natural growth rate of preys
    beta = 0.023  # death rate per encounter of preys due to predation
    gamma = 0.07  # natural growth rate of predators
    delta = 0.0021  # natural death rate of predators in the absence of food

    iterations = 100  # number of iterations
