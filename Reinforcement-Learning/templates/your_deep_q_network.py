# -*- coding: utf-8 -*-
from pyqlearning.deepqlearning.deep_q_network import DeepQNetwork


class YourDeepQNetwork(DeepQNetwork):
    '''
    Deep Q-Network to solive your problem.
    '''

    def inference(self, state_arr, limit=1000):
        '''
        Infernce.
        
        Args:
            state_arr:    `np.ndarray` of state.
            limit:        The number of inferencing.
        
        Returns:
            `list of `np.ndarray` of an optimal route.
        '''
        # Your concrete functions.
        pass

    def extract_possible_actions(self, state_arr):
        '''
        Extract possible actions.

        This method is overrided.

        Args:
            state_arr:  `np.ndarray` of state.
        
        Returns:
            `np.ndarray` of actions.
        '''
        # Your concrete functions.
        pass

    def observe_reward_value(self, state_arr, action_arr):
        '''
        Compute the reward value.

        This method is overrided.

        Args:
            state_arr:              `np.ndarray` of state.
            action_arr:             `np.ndarray` of action.
        
        Returns:
            Reward value.
        '''
        # Your concrete functions.
        pass

    def update_state(self, state_arr, action_arr):
        '''
        Update state.
        
        This method is overrided.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.
            action_arr:   `np.ndarray` of action in `self.t`.
        
        Returns:
            `np.ndarray` of state in `self.t+1`.
        '''
        # Your concrete functions.
        pass

    def check_the_end_flag(self, state_arr):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        This method is overrided.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.

        Returns:
            bool
        '''
        # Your concrete functions.
        pass
