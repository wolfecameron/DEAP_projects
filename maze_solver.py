import numpy as np
import random 
from deap import algorithms, base, creator, tools

#this script uses DEAP evolutionary algorithm to solve a maze


#maze can be defined in this matrix
maze = np.matrix([[0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,1,1,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,1,0,0,0,0,0],
                  [0,0,0,0,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0]])

				  
#calculates final position based on list of moves by the individual

'''
Note:
[0,0] -> South
[1,0] -> East
[0,1] -> North
[1,1] -> West
'''

def evalMatrix(individual):
    y = 1
    x = 1
    for i in range(0,29,2):
        if individual[i:i+2] == [0,0]:
            z = y-1
            if maze[z,x] == 0:
                y = y
            if maze[z,x] == 1:
                y = z
        if individual[i:i+2] == [1,0]:
            z = x+1
            if maze[y,z] == 0:
                x = x
            if maze[y,z] == 1:
                x = z
        if individual[i:i+2] == [0,1]:
            z = y+1
            if maze[z,x] == 0:
                y= y
            if maze[z,x] == 1:
                y = z
        if individual[i:i+2] == [1,1]:
            z = x-1
            if maze[y,z] == 0:
                x = x 
            if maze[y,z] == 1:
                x = z
    fitness = np.sqrt((((5-y)**2))+(((8-x)**2)))
    return (1/(fitness+.01),)


creator.create("FitnessMax", base.Fitness, weights = (+1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)
tb = base.Toolbox()
tb.register("bit", random.randint,0,1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n=30)
tb.register("population", tools.initRepeat, list, tb.individual, n = 400)
tb.register("evaluate", evalMatrix)
tb.register("mate", tools.cxTwoPoint)
tb.register("mutate", tools.mutFlipBit, indpb = 0.05)
tb.register("select", tools.selTournament, tournsize = 5)
tb.register("map", map)

cxpb , mutpb, ngen = .05, .05, 400
pop = tb.population()
pop = algorithms.eaSimple(pop,tb,cxpb,mutpb,ngen)

final_ind = pop[0][0]
print final_ind