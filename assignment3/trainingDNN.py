import argparse
from squadro_state import SquadroState
import numpy as np
from training_with_itself import training, save_new_model

text = 'dummyDNN'
# text = 'model200neurons_7layers'
model_path     = 'model/{}.pt'.format(text)
new_model_path = 'model/{}_new.pt'.format(text)

"""
Runs the game
"""
def main(agent_0, agent_1, first):
    init = 1
    iterations = 0
    all_results = None
    

    
    while True:
        (results, winner) = game(agent_0, agent_1, first, validation=0)
        new_result = np.append(results, np.transpose(2 * np.abs(np.transpose(np.reshape(results[:,0], newshape=[-1,1])) + np.transpose(winner * np.ones((np.size(results, 0), 1))) -1) - 1), 1)
        all_results = new_result if init == 1 else np.append(all_results, new_result, 0)
        init += 1
        
        if init % 10 == 0:
            init = 1
            iterations += 1
            np.save('data/{}_{}'.format(text, iterations), all_results)
            
            training(all_results, model_path, new_model_path)
            
            # Validation
            victory = 0
            n = 50
            for i in range(n):
                (results, winner) = game(agent_0, agent_1, first, validation=1)
                victory += winner # number of victories for player 1
            if victory / n > 0.55:
                # model validated, replaced with new one
                save_new_model(model_path, new_model_path)
            print(victory / n)
            
            #evaluate_model()

def game(agent_0, agent_1, first, validation):
    
    results = 0
    init = 1

    # Initialisation
    cur_state = SquadroState()
    if first != -1:
        cur_state.cur_player = first
    agents = [getattr(__import__(agent_0), 'MyAgent')(), getattr(__import__(agent_1), 'MyAgent')()]
    agents[0].set_id(0)
    agents[1].set_id(1)
    if validation: # Use different models during validation phase
        agents[1].set_model_path(new_model_path)
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-ai0", help="path to the ai that will play as player 0")
	parser.add_argument("-ai1", help="path to the ai that will play as player 1")
	parser.add_argument("-f", help="indicates the player (0 or 1) that plays first; random otherwise")
	args = parser.parse_args()

	ai0 = args.ai0 if args.ai0 != None else "human_agent"
	ai1 = args.ai1 if args.ai1 != None else "human_agent"
	first = int(args.f) if args.f != None else -1

	main(ai0, ai1, first)
