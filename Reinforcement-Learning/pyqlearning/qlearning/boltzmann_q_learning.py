#!/user/bin/env python
# -*- coding: utf-8 -*-
import random
import math
from pyqlearning.q_learning import QLearning


class BoltzmannQLearning(QLearning):
    '''
    Q-Learning with Boltzmann distribution.

    '''

    # Time rate.
    __time_rate = 0.001

    def get_time_rate(self):
        '''
        getter
        Time rate.
        '''
        if isinstance(self.__time_rate, float) is False:
            raise TypeError("The type of __time_rate must be float.")

        if self.__time_rate <= 0.0:
            raise ValueError("The value of __time_rate must be greater than 0.0")

        return self.__time_rate

    def set_time_rate(self, value):
        '''
        setter
        Time rate.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __time_rate must be float.")

        if value <= 0.0:
            raise ValueError("The value of __time_rate must be greater than 0.0")

        self.__time_rate = value

    time_rate = property(get_time_rate, set_time_rate)

    def select_action(self, state_key, next_action_list):
        '''
        Select action by Q(state, action).
        
        Concreat method for boltzmann distribution.

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is 0, all action should be possible.

        Retruns:
            The key of action.

        '''
        next_action_b_list = self.__calculate_boltzmann_factor(state_key, next_action_list)

        if len(next_action_b_list) == 1:
            return next_action_b_list[0][0]

        next_action_b_list = sorted(next_action_b_list, key=lambda x: x[1])

        prob = random.random()
        i = 0
        while prob > next_action_b_list[i][1] + next_action_b_list[i + 1][1]:
            i += 1
            if i + 1 >= len(next_action_b_list):
                break

        max_b_action_key = next_action_b_list[i][0]
        return max_b_action_key

    def __calculate_sigmoid(self):
        '''
        Function of temperature.

        Returns:
            Sigmoid.

        '''
        sigmoid = 1 / math.log(self.t * self.time_rate + 1.1)
        return sigmoid

    def __calculate_boltzmann_factor(self, state_key, next_action_list):
        '''
        Calculate boltzmann factor.

        Args:
            state_key:              The key of state.
            next_action_list:       The possible action in `self.t+1`.
                                    If the length of this list is 0, all action should be possible.

        Returns:
            [(`The key of action`, `boltzmann probability`)]
        '''
        sigmoid = self.__calculate_sigmoid()
        parent_list = [(action_key, math.exp(self.extract_q_dict(state_key, action_key) / sigmoid)) for action_key in next_action_list]
        parent_b_list = [parent[1] for parent in parent_list]
        next_action_b_list = [(action_key, child_b / sum(parent_b_list)) for action_key, child_b in parent_list]
        return next_action_b_list
