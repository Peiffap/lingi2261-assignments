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
        ll = []
        for k in range(state.npacs):
            l = []
            (i, j) = state.pac_list[k]
            l.append((i,j))
            if i > 0 and state.grid[i-1][j] != 'x':
                l.append((i-1,j))
            if (i < state.nbr - 1) and state.grid[i+1][j] != 'x':
                l.append((i+1,j))
            if j > 0 and state.grid[i][j-1] != 'x':
                l.append((i,j-1))
            if (j < state.nbc - 1) and state.grid[i][j+1] != 'x':
                l.append((i,j+1))
            ll.append(l)
        
        def rec(ll, new_list, out_list, k):
            if k == len(ll):
                out_list.append(new_list.copy())
                new_list.pop(-1)
                return
            for i in range(len(ll[k])):
                new_list.append(ll[k][i])
                rec(ll, new_list, out_list, k+1)
            if k != 0:
                new_list.pop(-1)
                
        out_list = []
        rec(ll, [], out_list, 0)
        out_list.pop(0)
        
        def my_copy_list(state):
            newgrid = []
            for i in range(state.nbr):
                newgrid.append([" "] * state.nbc)
                for j in range(state.nbc):
                    newgrid[i][j] = state.grid[i][j]
            return newgrid
        
        for k in range(len(out_list)):
            if len(out_list[k]) != len(set(out_list[k])):
                continue
            newgrid = my_copy_list(state)
            nfoods = state.nfoods
            for (i,j) in state.pac_list:   # Remove the previous pacmen
                newgrid[i][j] = ' '
            for (i,j) in out_list[k]:      # Add the new pacmen
                if newgrid[i][j] == '@':
                    nfoods -= 1
                newgrid[i][j] = '$'
            newstate = State(newgrid, out_list[k], npacs=state.npacs, nfoods=nfoods)  # New state
            yield(0, newstate)             # Yield the action (0 because the paths are costless) and the state to the 'expand' method


    def goal_test(self, state):
        return state.nfoods == 0


###############
# State class #
###############
class State:
    def __init__(self, grid, pac_list=None, food_list=None, npacs=None, nfoods=None):
        self.nbr = len(grid)
        self.nbc = len(grid[0])
        self.grid = grid
        if pac_list == None:
            pac_list = []
            food_list = []
            nfoods = npacs = 0
            for i in range(self.nbr):
                for j in range(self.nbc):
                    if grid[i][j] == '$':
                        pac_list.append((i,j))
                        npacs += 1
                    elif grid[i][j] == '@':
                        food_list.append((i,j))
                        nfoods += 1
        self.pac_list = pac_list
        self.food_list = food_list
        self.npacs = npacs
        self.nfoods = nfoods
                        
            

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
        for i in range(self.nbr):                                            # Compare each tile one-by-one
            for j in range(self.nbc):
                if self.grid[i][j] != other_state.grid[i][j]:
                    return False
        return True

    def __hash__(self):
        ctr = 0
        for i in range(self.nbr):
            for j in range(self.nbc):
                if self.grid[i][j] == "$":
                    ctr += (self.nbr * i + j) * 4 + 1
                elif self.grid[i][j] == "@":
                    ctr += (self.nbr * i + j) * 4 + 2
        return ctr



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
    h = 0
    return h


#####################
# Launch the search #
#####################
grid_init = readInstanceFile(sys.argv[1])
init_state = State(grid_init)
print(init_state)

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

print("nb nodes explored = ",nbExploredNodes)
print("time : " + str(endTime - startTime))
