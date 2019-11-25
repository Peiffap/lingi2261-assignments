from squadro_state import SquadroState
import numpy as np
import torch

ai0 = 'contest_agent3'
ai1 = 'smart_agent'
# text = 'model200neurons_7layers'
model_path = 'model/{}.pt'.format(ai0)

first = 0

"""
Runs the game
"""
def main(agent_0, agent_1, first):
    victory = 0
    for i in range(1000):
        _, winner = game(agent_0, agent_1, first, i)
        victory += 1 - winner # number of victories for player 0 (main)
        print('Victory average for the DNN model VS smart model', i, ':', 100 * victory / (i+1), '%')

def game(agent_0, agent_1, first, i):
    
    results = 0
    init = 1

    # Initialisation
    cur_state = SquadroState()
    if first != -1:
        cur_state.cur_player = first
    agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    #if i == 0:
        #print('Network 0 (main) -------------------------------------------------------')
        #print_network(agents[0].deepnetwork)
        #print('Network 1 (other) -------------------------------------------------------')
        #print_network(agents[1].deepnetwork)

    last_action = None
    
    while not cur_state.game_over():
    
        # Make move
        cur_player = cur_state.get_cur_player()
        action = get_action_timed(agents[cur_player], cur_state.copy(), last_action)
    
        if cur_state.is_action_valid(action):
            cur_state.apply_action(action)
            last_action = action
        else:
            cur_state.set_invalid_action(cur_player)

    return (0, cur_player)

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 20)
	return action

def print_network(network):
    # Print model's state_dict
    mod_dict = network.state_dict()
    print("Model's state_dict:")
    for param_tensor in mod_dict:
        print(param_tensor, "\t", mod_dict[param_tensor].size())
        #print(mod_dict[param_tensor])
        print(torch.sum(mod_dict[param_tensor]))

if __name__ == "__main__":

	main(ai0, ai1, first)
