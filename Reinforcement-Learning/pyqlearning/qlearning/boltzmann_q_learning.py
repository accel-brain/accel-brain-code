# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
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
        if self.q_df is None or self.q_df.shape[0] == 0:
            return random.choice(next_action_list)

        next_action_b_df = self.__calculate_boltzmann_factor(state_key, next_action_list)

        if next_action_b_df.shape[0] == 1:
            return next_action_b_df["action_key"].values[0]

        prob = np.random.random()
        next_action_b_df = next_action_b_df.sort_values(by=["boltzmann_factor"])

        i = 0
        while prob > next_action_b_df.iloc[i, :]["boltzmann_factor"] + next_action_b_df.iloc[i + 1, :]["boltzmann_factor"]:
            i += 1
            if i + 1 >= next_action_b_df.shape[0]:
                break

        max_b_action_key = next_action_b_df.iloc[i, :]["action_key"]
        return max_b_action_key

    def __calculate_sigmoid(self):
        '''
        Function of temperature.

        Returns:
            Sigmoid.

        '''
        sigmoid = 1 / np.log(self.t * self.time_rate + 1.1)
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
        q_df = self.q_df[self.q_df.state_key == state_key]
        q_df = q_df[q_df.isin(next_action_list)]
        q_df["boltzmann_factor"] = q_df["q_value"] / sigmoid
        q_df["boltzmann_factor"] = q_df["boltzmann_factor"].apply(np.exp)
        q_df["boltzmann_factor"] = q_df["boltzmann_factor"] / q_df["boltzmann_factor"].sum()
        
        return q_df
