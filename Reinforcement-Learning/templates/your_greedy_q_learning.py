# -*- coding: utf-8 -*-
from pyqlearning.qlearning.greedy_q_learning import GreedyQLearning


class YourGreedyQLearning(GreedyQLearning):
    '''
    ε-greedy Q-Learning to solve your problem.
    '''

    def select_action(self, state_key, next_action_list):
        '''
        Select action by Q(state, action).

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is zero, all action should be possible.

        Returns:
            The key of action.

        '''
        pass

    def extract_possible_actions(self, state_key):
        '''
        Extract the list of the possible action in `self.t+1`.

        Abstract method for concreate usecases.

        Args:
            state_key       The key of state in `self.t+1`.

        Returns:
            `list` of the possible actions in `self.t+1`.

        '''
        pass

    def observe_reward_value(self, state_key, action_key):
        '''
        Compute the reward value.
        
        Args:
            state_key:              The key of state.
            action_key:             The key of action.
        
        Returns:
            Reward value.
        '''
        pass

    def update_state(self, state_key, action_key):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            state_key:    The key of state in `self.t`.
            action_key:   The key of action in `self.t`.
        
        Returns:
            The key of state in `self.t+1`.
        '''
        pass

    def check_the_end_flag(self, state_key):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_key:    The key of state in `self.t`.

        Returns:
            bool
        '''
        pass
