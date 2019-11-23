from agent import AlphaBetaAgent
from time import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import logging
import random
run_folder = './run/'

class DeepNetwork(nn.Module):
    def __init__(self):
        """
          nin: #input channels
          nout: #output channels
          ksize: kernel size (same for both dimensions)
        """
        super().__init__()
        
        nin = 10             # 10 inputs: 5 numbers for each player
        nout = 5             # 5 outputs: probability to choose one of the 5 actions
        hidden_layers = 200  # Size of the 3 hidden layers

        self.fc1 = nn.Linear(nin, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, hidden_layers)
        self.fc3 = nn.Linear(hidden_layers, hidden_layers)
        self.fcv = nn.Linear(hidden_layers, 1)
        self.fcp = nn.Linear(hidden_layers, nout)
        self.batch200 = nn.BatchNorm1d(200)
        self.batch5 = nn.BatchNorm1d(5)
                  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #print(x)
        #print(F.log_softmax(x))
        vh = F.relu(self.fcv(x))
        ph = F.relu(self.fcp(x))
        return (F.log_softmax(ph), vh)

"""
Contest agent
"""
class MyAgent(AlphaBetaAgent):

    def __init__(self):
        self.current_depth = 0
        self.action_size = 5
        self.max_depth = 9
        self.max_time = 0
        self.start_time = 0
        self.total_time = 0
        self.mcts = None

        self.deepnetwork = DeepNetwork()
        self.tensor_state = None

        #for param in self.deepnetwork.parameters():
        #    print(param.data)

        # create a stochastic gradient descent optimizer
        optimizer = optim.SGD(self.deepnetwork.parameters(), lr=0.01, momentum=0.9)
        # create a loss function
        criterion = nn.NLLLoss()

        print(self.deepnetwork)


    def get_name(self):
        return 'Group 13'
    
    
    """
    This is the smart class of an agent to play the Squadro game.
    """
    def get_action(self, state, last_action, time_left):
        self.last_action = last_action
        self.time_left = time_left
        self.current_depth = 0
        self.start_time = time()
        if self.total_time == 0:
            self.total_time = time_left;
        if time_left / self.total_time > 0.2:
            self.max_time = 0.03 * self.total_time
        else:
            self.max_time = 0.03 * self.total_time * (time_left / (0.2 * self.total_time))**2
        best_move = 1
        # print(time_left)
        
        root = Node(state)
        self.mcts = MCTS(root)
        
        #### MCTS
        value = self.evaluateLeaf(root, 0)
        n = 1
        while time() - self.start_time < self.max_time and n < 50:
            #print(time() - self.start_time)
            #best_move = minimax.search(state, self)
            logger_mcts.info('***************************')
            logger_mcts.info('****** SIMULATION %d ******', n)
            logger_mcts.info('***************************')
            self.simulate(state)
            n += 1
        #print("Finish")
        #print(self.current_depth)
        #print("Time elapsed during smart agent play:", time() - self.start_time)
        
        pi, values = self.getAV(1)
        
         #### pick the action (at random with prob = epsilon)
        epsilon = 0.03
        tau = 1 if random.uniform(0, 1) < epsilon else 0
        best_move, value = self.chooseAction(pi, values, tau)

        nextState, _ = self.takeAction(state, best_move)

        NN_value = -self.evaluate(nextState)[1]

        logger_mcts.info('ACTION VALUES...%s', pi)
        logger_mcts.info('CHOSEN ACTION...%d', best_move)
        logger_mcts.info('MCTS PERCEIVED VALUE...%f', value)  # Value estimated by MCTS: Q = W/N (average of the all the values along the path)
        logger_mcts.info('NN PERCEIVED VALUE...%f', NN_value) # Value estimated by the Neural Network (only for the next state)

        #print(best_move, pi, value, NN_value)
      
        l1 = [state.get_pawn_advancement(self.id, pawn) for pawn in [0, 1, 2, 3, 4]]
        l2 = [state.get_pawn_advancement(1 - self.id, pawn) for pawn in [0, 1, 2, 3, 4]]
        print('{} {} {}'.format(l1, l2, best_move))
        
        return best_move
  
    
    def simulate(self, state):
      
        logger_mcts.info('ROOT NODE...')
        render(self.mcts.root.state, logger_mcts)
        logger_mcts.info('CURRENT PLAYER...%d', self.mcts.root.playerTurn)
        
        ##### MOVE THE LEAF NODE
        # Breadcrumbs = path from root to leaf
        leaf, done, breadcrumbs = self.mcts.moveToLeaf(self)
        render(leaf.state, logger_mcts)

        ##### EVALUATE THE LEAF NODE with deep neural network + add edges to leaf node
        value = self.evaluateLeaf(leaf, done)

        ##### BACKFILL THE VALUE THROUGH THE TREE
        self.mcts.backFill(leaf, value, breadcrumbs)


    def evaluateLeaf(self, leaf, done):

        logger_mcts.info('------EVALUATING LEAF------')

        value = 1

        if done == 0:
    
            probs, value = self.evaluate(leaf.state)
            print(probs)
            allowedActions = leaf.state.get_current_player_actions()
            logger_mcts.info('PREDICTED VALUE FOR %d: %f', leaf.playerTurn, value)

            probs = probs[allowedActions]

            # Add node in tree
            for idx, action in enumerate(allowedActions):
                newState, _ = self.takeAction(leaf.state, action)
                node = Node(newState)
                #if newState.id not in self.mcts.tree:
                #self.mcts.addNode(node)
                logger_mcts.info('added node......p = %f', probs[idx])
                #else:
                #    node = self.mcts.tree[newState.id]
                 #   logger_mcts.info('existing node...%s...', node.id)

                newEdge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append((action, newEdge))

        return value
    
    
    def getAV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.action_size, dtype=np.integer)
        values = np.zeros(self.action_size, dtype=np.float32)
        
        for action, edge in edges:
            pi[action] = pow(edge.stats['N'], 1/tau)
            values[action] = edge.stats['Q']
        pi = pi / (np.sum(pi) * 1.0)
        return pi, values
      
      
    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0] # if several states have the same prob
        else:
            action_idx = np.random.multinomial(1, pi)
            action = np.where(action_idx==1)[0][0] # random action

        value = values[action]

        return action, value
        
    
    def takeAction(self, state, a):
      
        newState = state.copy()
        newState.apply_action(a)

        value = 0
        
        done = 1 if self.cutoff(newState, self.max_depth) else 0

        return (newState, done) 


    """
    The successors function must return (or yield) a list of
    pairs (a, s) in which a is the action played to reach the
    state s.
    """
    def successors(self, state):
        actions = state.get_current_player_actions()
        for a in actions:
            s = state.copy()
            s.apply_action(a)
            yield (a, s)


    """
    The cutoff function returns true if the alpha-beta/minimax
    search has to stop and false otherwise.
    """
    def cutoff(self, state, depth):
        return depth > self.current_depth or state.game_over_check() or time() - self.start_time > self.max_time


    """
    The evaluate function must return an integer value
    representing the utility function of the board.
    """
    def evaluate(self, state):
        l1 = [state.get_pawn_advancement(self.id, pawn) for pawn in [0, 1, 2, 3, 4]]
        l2 = [state.get_pawn_advancement(1 - self.id, pawn) for pawn in [0, 1, 2, 3, 4]]
        x = torch.FloatTensor(l1 + l2)
        ph, vh = self.deepnetwork(x)
        ph = ph.data.numpy()
        vh = np.float(vh.data.numpy())
        return (ph, vh) # Deep neural network evaluation
        #return (np.array([0.2, 0.2, 0.2, 0.2, 0.2]), sum(l2)-sum(l1)) # basic eval function



# Source: https://github.com/AppliedDataSciencePartners/DeepReinforcementLearning/

class MCTS():

    def __init__(self, root):
        self.root = root
        self.cpuct = 1
        #self.tree = {}
    
    #def __len__(self):
    #    return len(self.tree)


    def moveToLeaf(self, player):

        logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        currentNode = self.root

        done = 0

        while not currentNode.isLeaf():

            logger_mcts.info('PLAYER TURN...%d', currentNode.playerTurn)
        
            maxQU = -9999

            # Choose randomly at 20% for the root node (ONLY for the training)
            epsilon = 0.2 if currentNode == self.root else 0
            nu = np.random.dirichlet([0.8] * len(currentNode.edges)) if currentNode == self.root else [0] * len(currentNode.edges)

            Nb = 0
            for action, edge in currentNode.edges:
                Nb = Nb + edge.stats['N']

            for idx, (action, edge) in enumerate(currentNode.edges):

                U = self.cpuct * ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats['N'])
                    
                Q = edge.stats['Q']

                logger_mcts.info('action: %d (%d)... N = %d, P = %f, nu = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
                    , action, action % 7, edge.stats['N'], np.round(edge.stats['P'],6), np.round(nu[idx],6), ((1-epsilon) * edge.stats['P'] + epsilon * nu[idx] )
                    , np.round(edge.stats['W'],6), np.round(Q,6), np.round(U,6), np.round(Q+U,6))

                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationAction = action
                    simulationEdge = edge

            logger_mcts.info('action with highest Q + U...%d', simulationAction)

            newState, done = player.takeAction(currentNode.state, simulationAction) # the value of the newState from the POV of the new playerTurn
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)

        logger_mcts.info('DONE...%d', done)

        return currentNode, done, breadcrumbs


    def backFill(self, leaf, value, breadcrumbs):
        logger_mcts.info('------DOING BACKFILL------')

        currentPlayer = leaf.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.playerTurn
            direction = 1 if playerTurn == currentPlayer else -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

            logger_mcts.info('updating edge with value %f for player %d... N = %d, W = %f, Q = %f'
                , value * direction
                , playerTurn
                , edge.stats['N']
                , edge.stats['W']
                , edge.stats['Q']
                )

            render(edge.outNode.state, logger_mcts)

    #def addNode(self, node):
     #   self.tree[node.id] = node



#### Node class
class Node():

    def __init__(self, state):
        self.state = state
        self.playerTurn = state.get_cur_player()
        self.edges = []

    def isLeaf(self):
        return len(self.edges) == 0




#### Edge class
class Edge():

    def __init__(self, inNode, outNode, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.playerTurn = inNode.playerTurn
        self.action = action

        self.stats =  {
                    'N': 0,
                    'W': 0,
                    'Q': 0,
                    'P': prior,
                }
              
        
        
##############################################################################      
def setup_logger(name, log_file, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


### SET all LOGGER_DISABLED to True to disable logging
### WARNING: the mcts log file gets big quite quickly

LOGGER_DISABLED = {
'main':False
, 'memory':False
, 'tourney':False
, 'mcts':False
, 'model': False}


logger_mcts = setup_logger('logger_mcts', 'logs/logger_mcts.log')
logger_mcts.disabled = LOGGER_DISABLED['mcts']

def render(state, logger):
    logger.info(state.cur_pos)
    logger.info('--------------')