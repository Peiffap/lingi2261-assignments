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
        positions = []  # Create a list of possible positions (successors) based on the current position (state)
        for pos in [(-2,-1), (-1,-2), (2,-1), (-1,2), (2,1), (1,2), (-2,1), (1,-2)]:    # The Knight moves following an 'L' shape
            x = state.x + pos[1]                                                        # Next x (horizontal) position
            y = state.y + pos[0]                                                        # Next y (vertical) position
            if (x < state.nCols and x >= 0 and y < state.nRows and y >= 0 and state.grid[y][x] != "♞"): # If the next position is in the board and not yet visited
                 positions.append((y,x))                                                                # Add the position to the list of successors
                 
        """
            border(position)
            
        computes the distance from the argument position to the nearest border
        """
        def border(position):
            return min(position[0]**2 + position[1]**2, position[0]**2 + (state.nCols - position[1] - 1)**2, (state.nRows - position[0] - 1)**2 + position[1]**2, (state.nCols - position[1] - 1)**2 + (state.nRows - position[0] - 1)**2)
        
        # We want to reach first the successors closest to a border.
        # Thus, we sort the successors of "state" following their distance from a border.
        # It is sorted in descending order since the yield method gives the states one-by-one, 
        # it gives thus the state closest to a border in the end.
        # This state is then on top of the queue and checked first when 'frontier' is a LIFO queue.
        # However, without changing the methods in search.py, we cannot sort the successors 
        # for both the depth and breadth search at the same time:
        # for the BFS (FIFO queue), the successors are sorted following the wrong
        # (descending) order in 'frontier' and will thus begin to check the states further away from the border.
        # Since DFS is faster without sort, we have chosen to sort in the right order for this search.
        positions = sorted(positions, key=lambda pos: border(pos), reverse=True)
        for pos in positions:
            x = pos[1]
            y = pos[0]
            newstate = State((state.nCols, state.nRows), (y, x), state.n + 1, state.grid)  # New state with the new initial
            newstate.grid[state.y][state.x] = "♞"                                          # position and n+1 visited tiles
            yield(0, newstate)  # Yield the action (0 because the paths are costless) and the state to the 'expand' method
            

    def goal_test(self, state):                      # Check if the state is the goal:
        return state.n == state.nRows * state.nCols  # n = nRows * nCols when all the tiles of a state have been visited



###############
# State class #
###############

class State:
    def __init__(self, shape, init_pos, num=1, grid=None):  # New params: num (number of visited tiles)
        self.nCols = shape[0]                               #             grid (grid of the ancestor, if present)
        self.nRows = shape[1]
        self.grid = []
        for i in range(self.nRows):
            self.grid.append([" "] * self.nCols)
        if grid != None:
            for i in range(self.nRows):
                for j in range(self.nCols):
                    self.grid[i][j] = grid[i][j]
        self.grid[init_pos[0]][init_pos[1]] = "♘"
        self.x = init_pos[1]  # y (vertical) coordinate of the initial tile
        self.y = init_pos[0]  # x (horizontal) coordinate of the initial tile
        self.n = num          # number of visited tiles

    def __str__(self):  # To string method
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
    # Compare the value of both grids, as well as their symmetries
    def __eq__(self, other):
        if (self.x != other.x) or (self.y != other.y) or (self.n != other.n):  # Quick check before entering the double for loop
            return False
        for i in range(self.nRows):                                            # Compare each tile one-by-one
            for j in range(self.nCols):
                if (self.grid[i][j] != other.grid[i][j] and (self.nRows == self.nCols and self.grid[i][j] != other.grid[j][i]) and self.grid[i][j] != other.grid[self.nRows-1-i][self.nCols-1-j] and self.grid[i][j] != other.grid[self.nRows-1-i][j] and self.grid[i][j] != other.grid[i][self.nCols-1-j]):
                    return False
        return True
    
    # Hash function required to make this class comparable: convert an object into an integer.
    # The hash of the grid is sufficient to completely describe this class.
    # Each tile can have 3 values, we thus store each tile on 2 bits (0 for " ", 1 for "♘" and 2 for "♞").
    # Then, we just sum all the values for each tile in order to get a unique hash associated to a specific grid.
    # For handling symmetrical states, we have to only implement a hash based on n since 2 
    # symmetrical grids need to have the same hash.
    def __hash__(self):
        ctr = 0
        for i in range(self.nRows):
            for j in range(self.nCols):
                if self.grid[i][j] == "♘":
                    ctr += (self.nRows * i + j) * 4 + 1
                elif self.grid[i][j] == "♞":
                    ctr += (self.nRows * i + j) * 4 + 2
        return ctr
        #return hash(self.n)

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
    '''
    for n in path:
        print(n.state)  # assuming that the __str__ function of state outputs the correct format
        print()
    '''
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
