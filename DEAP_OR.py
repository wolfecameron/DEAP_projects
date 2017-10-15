import numpy as np
import random
from deap import algorithms, base, creator, tools
#this script uses DEAP library to train a fully-connected neural network to solve the 'OR' problem
#OR definition: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->1


#evaluates each set of weights to see how it performs
#assigns fitness value to each set of weights

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

    #holds each output of the neural network
    inputMatrix = np.dot(possibleValues,weightsMatrix)


    #checks output of neural newtork against ideal output
    fit1 = (inputMatrix[0,0])**2
    fit2 = (inputMatrix[1,0]-1)**2
    fit3 = (inputMatrix[2,0]-1)**2
    fit4 = (inputMatrix[3,0]-1)**2

    fitness = fit1 + fit2 + fit3 + fit4
    return fitness,

#configures all features for DEAP library
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

#creates initial population and evolves it
cxpb , mutpb, ngen = .05, .05, 400
pop = tb.population()
pop = algorithms.eaSimple(pop,tb,cxpb,mutpb,ngen)

#selects first set of weights from last generation
weightsList = pop[0][0]

#uses derived weights to evaluate OR
def defineOR(x,y):
    #checks to ensure proper input for function
    if( (x!=0 and x!=1) or (y!=0 and y!=1)):
        return "Please input either a 1 or 0 for x and y."


    w = 0 #bias, not needed for OR problem
    w1= weightsList[0]
    w2 = weightsList[1]

    #sets input values into neural network
    inputMatrix = np.matrix([[x,y]])

    #creates weights by which inputs will be multiplied
    weightsMatrix = np.matrix([[w1],
                            [w2]])

    #holds the output of the neural network
    outputMatrix = np.dot(inputMatrix,weightsMatrix)
    outputValue = outputMatrix[0,0] + w

    if outputValue >= .5:
        return True
    if outputValue < .5:
        return False

#tests output
print(defineOR(0,0))
print(defineOR(1,0))
print(defineOR(0,1))
print(defineOR(1,1))
print(defineOR(10000,-.99)) #checks to make sure you must input 1 or 0

'''
should yield:
0
1
1
1
'''
