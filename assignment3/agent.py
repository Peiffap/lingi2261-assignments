from state import State

"""
Class that represents a agent.
"""
class Agent():

    """
    Compute the action to perfom on the current state
    of the game. The must be compute in at most time_left
    seconds.

    state: the current state
    time_left: the number of second left

    """
    def get_action(self, state, last_action, time_left):
        abstract

    def get_name(self):
        return 'student agent'

    """
    Set the id of the agent in the game. In a two player 
    game it will be either 0 if we play first of 1 otherwise.
    """
    def set_id(self, id):
        self.id = id

