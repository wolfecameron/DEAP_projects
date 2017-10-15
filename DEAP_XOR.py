import numpy as np
import random
from deap import algorithms, base, creator, tools
#this script uses DEAP library to train a fully-connected neural network to solve the 'XOR' problem
#XOR definition: (0,0)->0, (1,0)->1, (0,1)->1, (1,1)->0

#activation function for neural network
def sigmoid(z,deriv):
    if(deriv):
        return z(1-z)

    return 1/(1+np.exp(-z))

#evaluates each set of wiehgts based on performance for XOR
def evalWeights(individual):
    #holds all possible inputs (including bias)
    possibleValues = np.matrix([[0,1,0,1],
                               [0,0,1,1],
                               [1,1,1,1]])

    idealMatrix = np.matrix([[0],[1], [1], [0]])

    #holds weights by which inputs will be multiplied
    weightsMatrix = np.matrix([[individual[0],individual[2], individual[6]],
                           [individual[1],individual[3],individual[7]],
                          [0,0,1]])

    #multiplies inputs by weights to yield first hidden layer in neural network
    layer1 = sigmoid((np.dot(weightsMatrix,possibleValues)), False)

    #holds the second set of weights, by which hidden layer will be multiplied
    weightsMatrix2 = np.matrix([[individual[5], individual[4], individual[8]]])

    #holds output of neural networks
    outputMatrix = sigmoid(np.dot(weightsMatrix2,layer1), False)

    #compares ouput to ideal output to assign fitness
    fit1 = (outputMatrix[0,0])**2
    fit2 = (outputMatrix[0,1]-1)**2
    fit3 = (outputMatrix[0,2]-1)**2
    fit4 = (outputMatrix[0,3])**2

    fitness = fit1+fit2+fit3+fit4

    return (fitness,)


#configures all properties of DEAP library
creator.create("FitnessMin", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMin)
tb = base.Toolbox()
tb.register("floatAttribute", random.uniform,-1,1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.floatAttribute, n=9)
tb.register("population", tools.initRepeat, list, tb.individual, n = 200)
tb.register("evaluate", evalWeights)
tb.register("mate", tools.cxTwoPoint)
tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb = 0.05)
tb.register("select", tools.selTournament, tournsize = 3)
tb.register("map", map)

#creates initial population and evolves it based on fitness
cxpb , mutpb, ngen = .05, .05, 400
pop = tb.population()
pop = algorithms.eaSimple(pop,tb,cxpb,mutpb,ngen)

weightsList = pop[0][0]

def defineXOR(x,y):
    #checks to ensure proper input for function
    if( (x!=0 and x!=1) or (y!=0 and y!=1)):
        return "Please input either a 1 or 0 for x and y."

    #input values
    possibleValues = np.matrix([[x],
                                [y],
                               [1]])
    idealMatrix = np.matrix([[0],[1], [1], [0]])

    #first weights matrix
    weightsMatrix = np.matrix([[weightsList[0],weightsList[2], weightsList[6]],
                           [weightsList[1],weightsList[3],weightsList[7]],
                          [0,0,1]])

    #values for first hidden layer
    layer1 = sigmoid((np.dot(weightsMatrix,possibleValues)),False)

    #second weights matrix
    weightsMatrix2 = np.matrix([[weightsList[5], weightsList[4], weightsList[8]]])

    #output value
    outputMatrix = sigmoid(np.dot(weightsMatrix2,layer1), False)

    if outputMatrix >= .5:
        return True
    if outputMatrix < .5:
        return False

print(defineXOR(0,0))
print(defineXOR(1,0))
print(defineXOR(0,1))
print(defineXOR(1,1))
print(defineXOR(.5,-150))

'''
ideal results:
0
1
1
0
'''
