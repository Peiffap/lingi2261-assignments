# -*-coding: utf-8 -*
'''NAMES OF THE AUTHOR(S): Gael Aglin     <gael.aglin@uclouvain.be>
                           Martin Braquet <martin.braquet@student.uclouvain.be>
                           Gilles Peiffer <gilles.peiffer@student.uclouvain.be>
'''
import time
import sys
from search import *

#################
# Problem class #
#################
class Knight(Problem):

    def successor(self, state):
        positions = []                     # Create a list of possible positions (successors) based on the current position (state)
        for pos in [(-2,-1), (-1,-2), (2,-1), (-1,2), (2,1), (1,2), (-2,1), (1,-2)]:    # The Kight moves following an 'L' shape
            x = state.x + pos[1]                                                        # Next x (horizontal) position
            y = state.y + pos[0]                                                        # Next y (vertical) position
            if (x < state.nCols and x >= 0 and y < state.nRows and y >= 0 and state.grid[y][x] != "♞"): # If the next position is in the board and not already visited
                 positions.append((y,x))                                                # Add the position in the list of successors
                 
        # nsucc computes the number of successors for all the possible successors of the current state
        def nsucc(position):
            # Initialize the counter (start at -1 because one of the next positions of the successor
            # is a "♘", and the computation is reduced if we do not add this fifth condition in the 'if' of line 31)
            ctr = -1                                                               
            for pos in [(-2,-1), (-1,-2), (2,-1), (-1,2), (2,1), (1,2), (-2,1), (1,-2)]:
                x = position[1] + pos[0]                                                   # Next x (horizontal) position
                y = position[0] + pos[1]                                                   # Next y (vertical) position
                if (x < state.nCols and x >= 0 and y < state.nRows and y >= 0 and state.grid[y][x] != "♞"):
                     ctr += 1
            return ctr
        
        # We want to reach first the successors with the lowest number of successors.
        # Thus, we sort the successors of "state" following the number of their own successors.
        # It is sorted in descending order since the yield method gives the states one-by-one, 
        # it gives thus the state with the lowest successors in the end. This state is then on top of 
        # queue and checked first when 'frontier' is LIFO queue.
        # However, without changing the methods in search.py, we cannot sort the successors 
        # for both the depth and breadth search at the same time:
        # for the breadth search (FIFO queue), the successors are sorted following the wrong
        # (descending) order in 'frontier' and will thus begin to check the states with the higher number
        # of successors. Since the depth search is faster without sort, we have chosen to sort in the right order for this search.
        positions = sorted(positions, key=lambda pos: nsucc(pos), reverse=True)
        for pos in positions:
            x = pos[1]
            y = pos[0]
            newstate = State((state.nCols, state.nRows), (y, x), state.n + 1, state.grid) # New state with the new initial
            newstate.grid[state.y][state.x] = "♞"                                                  # position and n+1 busy tiles
            yield(0, newstate)                  # Yield the action (0 because the paths are costless) and the state to the 'expand' method
            

    def goal_test(self, state):                               # Check if the state is the goal:
        return state.n == state.nRows * state.nCols           # n = nRows * nCols when all the tiles of a state are busy



###############
# State class #
###############

class State:
    def __init__(self, shape, init_pos, num=1, grid=None):       # New params: num (number of busy tiles)
        self.nCols = shape[0]                                    #             grid (grid of the ancestor, if present)
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

    def __str__(self):                                # To string method
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
    
    
    # Comparison of the State class: required in order to set the 'state' class as a key (for the 'closed' dictionary)
    def __eq__(self, other):
        if (self.x != other.x) or (self.y != other.y) or (self.n != other.n):  # Quick check before entering the double for loop
            return False 
        for i in range(self.nRows):                                            # Compare each tile one-by-one
            for j in range(self.nCols):
                if self.grid[i][j] != other.grid[i][j]:
                    return False
        return True
    
    # Hash function required to make this class comparable
    # Normally it should be the grid, but the grid is a list (mutable). So it cannot be used as a key for the dictionary.
    # Need to fix this, maybe with a tuple (immutable) containing a copy of the grid...
    def  __hash__(self):
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
    
    #for n in path:
        #print(n.state)  # assuming that the __str__ function of state outputs the correct format
        #print()
    
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
