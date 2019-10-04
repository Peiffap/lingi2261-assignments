# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin     <gael.aglin@uclouvain.be>
                           Martin Braquet <martin.braquet@student.uclouvain.be>
                           Gilles Peiffer <gilles.peiffer@student.uclouvain.be>
'''
import time
import sys
from search import *
import copy


#################
# Problem class #
#################
class Knight(Problem):

    def successor(self, state):
        #print(state)
        for pos in [(-2,-1), (-1,-2), (2,-1), (-1,2), (2,1), (1,2), (-2,1), (1,-2)]:    # TODO: update, sort the successors following a 
            x = state.x + pos[0]                                                        # good scheme (maybe based on the position of 
            y = state.y + pos[1]                                                        # the knight, towards the center?)
            #print(state.n)
            if (x < state.nCols and x >= 0 and y < state.nRows and y >= 0 and state.grid[y][x] != "♞"):
                 newstate = State((state.nCols, state.nRows), (y, x), state.n + 1, state.grid)
                 newstate.grid[state.y][state.x] = "♞"
                 yield (0, newstate)

    def goal_test(self, state):
        return state.n == state.nRows * state.nCols                          # n = 25 when all the tiles of a state are busy

###############
# State class #
###############

class State:
    def __init__(self, shape, init_pos, num=1, grid=None):
        self.nCols = shape[0]
        self.nRows = shape[1]
        self.grid = []
        for i in range(self.nRows):
            self.grid.append([" "] * self.nCols)
        if grid != None:
            for i in range(self.nRows):
                for j in range(self.nCols):
                    self.grid[i][j] = grid[i][j]
        self.grid[init_pos[0]][init_pos[1]] = "♘"
        self.x = init_pos[1]                          # y (vertical) coordinate of the initial tile
        self.y = init_pos[0]                          # x (horizontal) coordinate of the initial tile
        self.n = num                                  # number of busy tiles

    def __str__(self):
        nsharp = (2 * self.nCols) + (self.nCols // 5)
        s = "#" * nsharp
        s += "\n"
        for i in range(self.nRows):
            s = s + "#"
            for j in range(self.nCols):
                s = s + str(self.grid[i][j]) + " "
            s = s[:-1]
            s = s + "#"
            if i < self.nRows - 1:
                s = s + '\n'
        s += "\n"
        s += "#" * nsharp
        return s
    
    
    # Comparison of the State class. Source: https://stackoverflow.com/a/5824757
    def __eq__(self, other):
        if (self.x != other.x) or (self.y != other.y) or (self.n != other.n):  # Quick check before entering the double for loop
            return False 
        for i in range(self.nRows):                                        # Compare each tile one-by-one
            for j in range(self.nCols):
                if self.grid[i][j] != other.grid[i][j]:
                    return False
        return True
    
    def  __hash__(self):
        # Normally it should be the grid, but the grid is a list (mutable). So it cannot be used as a key for the dictionary.
        # Need to fix this, maybe with a tuple (immutable) containing a copy of the grid...
        return hash(self.nRows)  

##############################
# Launch the search in local #
##############################
# Use this block to test your code in local
# Comment it and uncomment the next one if you want to submit your code on INGInious
with open('instances.txt') as f:
    instances = f.read().splitlines()

for instance in instances:
    elts = instance.split(" ")
    shape = (int(elts[0]), int(elts[1]))
    init_pos = (int(elts[2]), int(elts[3]))
    init_state = State(shape, init_pos)

    problem = Knight(init_state)

    # example of bfs graph search
    startTime = time.perf_counter()
    node, nbExploredNodes = depth_first_tree_search(problem)
    endTime = time.perf_counter()

    # example of print
    path = node.path()
    path.reverse()

    print('Number of moves: ' + str(node.depth))
    for n in path:
        print(n.state)  # assuming that the __str__ function of state outputs the correct format
        print()
    print("nb nodes explored = ", nbExploredNodes)
    print("time : " + str(endTime - startTime))


'''
####################################
# Launch the search for INGInious  #
####################################
#Use this block to test your code on INGInious
shape = (int(sys.argv[1]),int(sys.argv[2]))
init_pos = (int(sys.argv[3]),int(sys.argv[4]))
init_state = State(shape, init_pos)

problem = Knight(init_state)

# example of bfs graph search
startTime = time.perf_counter()
node, nbExploredNodes = depth_first_tree_search(problem)
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
'''
