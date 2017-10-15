import numpy as np
import random
from deap import algorithms, base, creator, tools
#this script uses DEAP library to train a fully-connected neural network to solve the 'AND' problem
#AND definition: (0,0)->0, (1,0)->0, (0,1)->0, (1,1)->1

#evaluates each set of weights by evaluating neural network
#assigns fitness to each set of weights
def evalWeights(individual):
    possibleValues = np.matrix([[0,0],
                           [0,1],
                           [1,0],
                           [1,1]])
    idealMatrix = np.matrix([[0],
                        [0],
                        [0],
                        [1]])

    weightsMatrix = np.matrix([[individual[0]],
                           [individual[1]]])

    #holds all output values for neural network
    inputMatrix = np.dot(possibleValues,weightsMatrix)

    #checks output values against ideal values to assign fitness
    fit1 = (inputMatrix[0,0])**2
    fit2 = (inputMatrix[1,0])**2
    fit3 = (inputMatrix[2,0])**2
    fit4 = (inputMatrix[3,0]-1)**2

    fitness = fit1 + fit2 + fit3 + fit4

    return fitness,

#configures all settings for DEAP framework
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("bit", random.random,)
tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n=2)
tb.register("population", tools.initRepeat, list, tb.individual, n = 400)
tb.register("evaluate", evalWeights)
tb.register("mate", tools.cxTwoPoints)
tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb = 0.05)
tb.register("select", tools.selTournament, tournsize = 5)
tb.register("map", map)


#creates population and evolves it
cxpb , mutpb, ngen = .05, .05, 400
pop = tb.population()
pop = algorithms.eaSimple(pop,tb,cxpb,mutpb,ngen)

weightsList = pop[0][0]

#creates AND function by evaluating nerual network with derived weights
def defineAnd(x,y):
    #checks to ensure proper input for function
    if( (x!=0 and x!=1) or (y!=0 and y!=1)):
        return "Please input either a 1 or 0 for x and y."

    w = 0 #bias, not needed for AND problem...
    w1= weightsList[0]
    w2 = weightsList[1]

    #sets input values for neural network
    inputMatrix = np.matrix([[x,y]])

    #creates weight matrix for input values to be multiplied by
    weightsMatrix = np.matrix([[w1],
                               [w2]])

    #holds output value of the neural network
    outputMatrix = np.dot(inputMatrix,weightsMatrix)

    outputValue = outputMatrix[0,0] + w

    if outputValue >= .5:
        return True
    if outputValue < .5:
        return False

#tests function
print(defineAnd(0,0))
print(defineAnd(1,0))
print(defineAnd(0,1))
print(defineAnd(1,1))
print(defineAnd(.5,6))

'''
should yield:
0
0
0
1
'''
