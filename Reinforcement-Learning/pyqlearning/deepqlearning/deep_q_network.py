# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.deep_q_learning import DeepQLearning


class DeepQNetwork(DeepQLearning):
    '''
    Abstract base class to implement the Deep Q-Network(DQN).
    '''

    # ε-greedy rate.
    __epsilon_greedy_rate = 0.75

    def select_action(self, next_action_arr, next_q_arr):
        '''
        Select action by Q(state, action).

        Args:
            next_action_arr:        `np.ndarray` of actions.
                                    The shape is:(
                                        `batch size corresponded to each action key`, 
                                        `channel that is 1`, 
                                        `feature points1`, 
                                        `feature points2`
                                    )
            
            next_q_arr:             `np.ndarray` of Q-Values.

        Retruns:
            Tuple(`np.ndarray` of action., Q-Value)
        '''
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))
        if epsilon_greedy_flag is False:
            key = np.random.randint(low=0, high=next_action_arr.shape[0])
        else:
            key = next_q_arr.argmax()

        return next_action_arr[key], next_q_arr[key]

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
