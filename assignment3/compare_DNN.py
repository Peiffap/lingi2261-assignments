from squadro_state import SquadroState
import numpy as np
import torch

text = 'contest_agent3'
# text = 'model200neurons_7layers'
model_path     = 'model/{}.pt'.format(text)
other_model_path = 'model/{}_initialized.pt'.format(text) #'model/{}_old_for_test.pt'.format(text)

ai0 = text
ai1 = ai0
first = 0

"""
Runs the game
"""
def main(agent_0, agent_1, first):
    victory = 0
    for i in range(1000):
        (results, winner) = game(agent_0, agent_1, first, i)
        victory += 1 - winner # number of victories for player 0 (main)
        print('Victory average for the main model VS other model', i, ':', 100 * victory / (i+1), '%')

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
    agents[1].set_model_path(other_model_path)
    agents[0].epsilonMCTS = 0
    agents[1].epsilonMCTS = 0
    agents[0].epsilonMove = 0
    agents[1].epsilonMove = 0
    if i == 0:
        print('Network 0 (main) -------------------------------------------------------')
        print_network(agents[0].deepnetwork)
        print('Network 1 (other) -------------------------------------------------------')
        print_network(agents[1].deepnetwork)

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
        
        if init:
            results = agents[cur_player].results
            init = 0
        else:
            results = np.append(results, agents[cur_player].results, 0)

    return (results, cur_player)

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 50)
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
