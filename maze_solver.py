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



#translates resulting bitList into directions
def parseResults(moveList):
	for i in range(0,len(moveList),2):
		if(moveList[i:i+2] == [0,0]):
			print "North"
		elif(moveList[i:i+2] == [1,0]):
			print "East"
		elif(moveList[i:i+2] == [0,1]):
			print "South"
		else:
			print "West"
				  
#calculates final position based on list of moves by the individual

'''
Note:
[0,0] -> North
[1,0] -> East
[0,1] -> South
[1,1] -> West
'''


MAX_MOVES = 13 
POP_SIZE = 200
INDPB = .1 #probability of X-over for cxUniform tool/mutation

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
    return fitness,
    #judges fitness based on final position in maze

	
	
creator.create("FitnessMax", base.Fitness, weights = (-1.0,))
creator.create("Individual", list, fitness = creator.FitnessMax)
tb = base.Toolbox()
tb.register("bit", random.randint,0,1)
tb.register("individual", tools.initRepeat, creator.Individual, tb.bit, n=(MAX_MOVES*2))
tb.register("population", tools.initRepeat, list, tb.individual, n= POP_SIZE)
tb.register("evaluate", evalMatrix)
tb.register("mate", tools.cxTwoPoint)
tb.register("mutate", tools.mutFlipBit, indpb = INDPB)
tb.register("select", tools.selTournament, tournsize = 5)
tb.register("map", map)



cxpb , mutpb, ngen = .1, .1, 600
pop = tb.population()
pop = algorithms.eaSimple(pop,tb,cxpb,mutpb,ngen)

#selects the first individual in the final population
final_ind = pop[0][0]

#prints all moves made by this individual
parseResults(final_ind)
