from squadro_state import SquadroState
import numpy as np
import random
import argparse
from training import training, save_new_model, print_network

n_train = 2
n_valid = 2
verif_prob = 0.55

"""
Runs the game
"""
def main(agent):
    #all_results = np.load('data/{}.npy'.format(text))
    all_results = np.zeros((1,17))
    #np.save('data/{}.npy'.format(text), all_results)
    
    model_path     = 'model/{}.pt'.format(agent)
    new_model_path = 'model/{}_new.pt'.format(agent)
    
    while True:
        
        for i in range(n_train):
            print('Training:', i)
            (results, winner) = game(agent, 0, i, new_model_path)
            new_result = np.append(results, np.transpose(2 * np.abs(np.transpose(np.reshape(results[:,0], newshape=[-1,1])) + np.transpose(winner * np.ones((np.size(results, 0), 1))) -1) - 1), 1)
            #all_results = new_result #if all_results == None  else np.append(all_results, new_result, 0)
            all_results = np.append(all_results, new_result, 0)
            
        #np.save('data/{}.npy'.format(agent), all_results)
        
        training(all_results, model_path, new_model_path, agent)
        
        all_results = np.zeros((1,17)) 
        
        # Validation
        print("Validation...")
        victory = 0
        for i in range(n_valid):
            (results, winner) = game(agent, 1, i, new_model_path)
            victory += winner # number of victories for player 1
            print('Validation', i, ':', 100 * victory / (i+1), '%')
            if i > n_valid / 2 and victory / (i+1) < verif_prob - 0.1:
                break
        if victory / n_valid > verif_prob:
            # model validated, replaced with new one
            save_new_model(model_path, new_model_path)
            #print(victory / n)
            
            #evaluate_model()

def game(agent, validation, i, new_model_path):
    
    results = 0
    init = 1

    # Initialisation
    cur_state = SquadroState()
    cur_state.cur_player = 0 if i % 2 == 0 else 1
    agents = [getattr(__import__(agent), 'MyAgent')(), getattr(__import__(agent), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    
    #print_network(agents[0].deepnetwork)
    
    if validation: # Use different models during validation phase
        ind = 0 if i % 4 < 2 else 1 
        agents[ind].set_model_path(new_model_path)
        print('Main model:', ind, ', First player:', cur_state.cur_player)
        # Remove stochastic actions
        #agents[0].epsilonMCTS = 0
        #agents[1].epsilonMCTS = 0
        agents[0].epsilonMove = 0
        agents[1].epsilonMove = 0
        '''
        print('Current network model...............................................')
        print_network(agents[0].deepnetwork)
        print('New network model....................................................')
        print_network(agents[1].deepnetwork)
        '''
        
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
            
    #if validation:
    #    print(results)
    #    print(cur_player)
    return (results, cur_player)

"""
Get an action from player with a timeout.
"""
def get_action_timed(player, state, last_action):
	action = player.get_action(state, last_action, 50)
	return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ai", help="path")
    args = parser.parse_args()

    ai = args.ai if args.ai != None else "error"
    main(ai)
