# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.deep_q_learning import DeepQLearning


class DeepQNetwork(DeepQLearning):
    '''
    Abstract base class to implement the Deep Q-Network(DQN).
    '''

    # ε-greedy rate.
    __epsilon_greedy_rate = 0.75
    
    def get_epsilon_greedy_rate(self):
        ''' getter '''
        if isinstance(self.__epsilon_greedy_rate, float) is True:
            return self.__epsilon_greedy_rate
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    def set_epsilon_greedy_rate(self, value):
        ''' setter '''
        if isinstance(value, float) is True:
            self.__epsilon_greedy_rate = value
        else:
            raise TypeError("The type of __epsilon_greedy_rate must be float.")

    epsilon_greedy_rate = property(get_epsilon_greedy_rate, set_epsilon_greedy_rate)

    def select_action(self, state_arr, next_action_arr):
        '''
        Select action by Q(state, action).

        Args:
            state_arr:              `np.ndarray` of state.
            next_action_arr:        `np.ndarray` of actions.
                                    The shape is:(
                                        `batch size corresponded to each action key`, 
                                        `channel that is 1`, 
                                        `feature points1`, 
                                        `feature points2`
                                    )
        Retruns:
            `np.ndarray` of action
        '''
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))
        if epsilon_greedy_flag is False:
            action_key = np.random.randint(low=0, high=next_action_arr.shape[0])
            return next_action_arr[action_key]
        else:
            return self.predict_next_action(state_arr, next_action_arr)
