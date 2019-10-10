# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin     <gael.aglin@uclouvain.be>
                           Martin Braquet <martin.braquet@student.uclouvain.be>
                           Gilles Peiffer <gilles.peiffer@student.uclouvain.be>
'''
import time
from search import *


#################
# Problem class #
#################
class Pacmen(Problem):

    def successor(self, state):
        pass

    def goal_test(self, state):
        pass


###############
# State class #
###############
class State:
    def __init__(self, grid):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid

    def __str__(self):
        nsharp = self.nbc * 2 + 3
        s = "#" * nsharp
        s += '\n'
        for i in range(0, self.nbr):
            s += "# "
            for j in range(0, self.nbc):
                s += str(self.grid[i][j]) + " "
            s += "#"
            if i < self.nbr:
                s += '\n'
        s += "#" * nsharp
        return s

    def __eq__(self, other_state):
        pass

    def __hash__(self):
        pass



######################
# Auxiliary function #
######################
def readInstanceFile(filename):
    lines = [[char for char in line.rstrip('\n')[1:][:-1]] for line in open(filename)]
    lines = lines[1:len(lines) - 1]
    n = len(lines)
    m = len(lines[0])
    grid_init = [[lines[i][j] for j in range(1, m, 2)] for i in range(0, n)]
    return grid_init


######################
# Heuristic function #
######################
def heuristic(node):
    h = 0.0
    # ...
    # compute an heuristic value
    # ...
    return h


#####################
# Launch the search #
#####################
grid_init = readInstanceFile(sys.argv[1])
init_state = State(grid_init)

problem = Pacmen(init_state)

startTime = time.perf_counter()
node, nbExploredNodes = astar_graph_search(problem,heuristic)
endTime = time.perf_counter()

# example of print
path = node.path()
path.reverse()

print('Number of moves: ' + str(node.depth))
for n in path:
    print(n.state)  # assuming that the __str__ function of state outputs the correct format
    print()

print("nb nodes explored = ",nb)
print("time : " + str(endTime - startTime))
