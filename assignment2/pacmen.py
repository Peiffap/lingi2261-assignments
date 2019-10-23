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
        states = [("t", state)]
        # Iterating over the pacmen
        
        def findpac(state):
            pac_list = []
            for i in range(state.nbr):
                for j in range(state.nbc):
                    if state.grid[i][j] == '$':
                        pac_list.append((i,j))
            return pac_list
        
        pac_list = findpac(state)
        
        for pacman in pac_list:
            possibilities = []
            # For each pacman, we iterate other the list of states left by the previous pacman, this list is only composed of the initial state for the first pacman
            for s in states:
                # This helps define the different directions in which the pacmen can move, they always move, because not moving is rarely optimal
                for move in [-1, 1]:
                    # Checking if the next positions are admissible
                    if 0 <= pacman[0] + move < s[1].nbr and s[1].grid[pacman[0]+move][pacman[1]] != "x" and s[1].grid[pacman[0]+move][pacman[1]] != "$":
                        newgrid = [row[:] for row in s[1].grid]
                        newgrid[pacman[0]][pacman[1]] = " "
                        newgrid[pacman[0]+move][pacman[1]] = "$"
                        possibilities.append(("t", State(newgrid)))
                    if 0 <= pacman[1] + move < s[1].nbc and s[1].grid[pacman[0]][pacman[1]+move] != "x" and s[1].grid[pacman[0]][pacman[1]+move] != "$":
                        newgrid = [row[:] for row in s[1].grid]
                        newgrid[pacman[0]][pacman[1]] = " "
                        newgrid[pacman[0]][pacman[1]+move] = "$"
                        possibilities.append(("t", State(newgrid)))
            states = possibilities[:] # The list of states from which pacmen will add their moves is updated to the list of states with all the states containing the different combinations of moves of the previous pacmen
        return states

    def goal_test(self, state):
        for i in range(state.nbr):
            for j in range(state.nbc):
                if state.grid[i][j] == "@":
                    return False
        return True


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
    
    # Two grids are equal if their elements are the same
    def __eq__(self, other_state):
        return self.grid == other_state.grid
    # Same logic as equal, the idea is that two states with similar grids will have the same hash.
    def __hash__(self):
        return hash(str(self.grid))



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

def heur(node):
    def manhattan(pos1, pos2):
        return abs(pos1[0]-pos2[0])+abs(pos1[1]-pos2[1])
    
    def findpf(state):
            pac_list = []
            food_list = []
            for i in range(state.nbr):
                for j in range(state.nbc):
                    if state.grid[i][j] == '$':
                        pac_list.append((i,j))
                    elif state.grid[i][j] == '@':
                        food_list.append((i,j))
            return food_list, pac_list
    
    (foods, pacmen) = findpf(node.state)
    
    l = [0]*len(foods)
    
    def closest_pac(food):
        tmp = manhattan(food,pacmen[0])
        closest = tmp
        for pac in pacmen[1:]:
            candidate = manhattan(pac,food)
            if candidate < closest:
                closest = candidate
        return closest
    
    for i in range(len(foods)):
        l[i] = closest_pac(foods[i])
    
    return max(l, default=0)

def zero(node):
    return 0
        

#####################
# Launch the search #
#####################
grid_init = readInstanceFile(sys.argv[1])
#grid_init = readInstanceFile("instances/i01")
init_state = State(grid_init)

problem = Pacmen(init_state)

startTime = time.perf_counter()
node, nbExploredNodes = astar_graph_search(problem,heur)
#node, nbExploredNodes = breadth_first_graph_search(problem)
endTime = time.perf_counter()

# example of print
path = node.path()
path.reverse()

print('Number of moves: ' + str(node.depth))
for n in path:
    print(n.state)  # assuming that the __str__ function of state outputs the correct format
    print()

print("nb nodes explored = ",nbExploredNodes)
print("time : " + str(endTime - startTime))
