# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
from pyqlearning.q_learning import QLearning


class BoltzmannQLearning(QLearning):
    '''
    Q-Learning with Boltzmann distribution.
    
    Boltzmann Q-Learning algorithm is based on Boltzmann action selection mechanism.

    References:
        - Agrawal, S., & Goyal, N. (2011). Analysis of Thompson sampling for the multi-armed bandit problem. arXiv preprint arXiv:1111.1797.
        - Bubeck, S., & Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. arXiv preprint arXiv:1204.5721.
        - Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).
        - Du, K. L., & Swamy, M. N. S. (2016). Search and optimization by metaheuristics (p. 434). New York City: Springer.
        - Kaufmann, E., Cappe, O., & Garivier, A. (2012). On Bayesian upper confidence bounds for bandit problems. In International Conference on Artificial Intelligence and Statistics (pp. 592-600).
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
        - Richard Sutton and Andrew Barto (1998). Reinforcement Learning. MIT Press.
        - Watkins, C. J. C. H. (1989). Learning from delayed rewards (Doctoral dissertation, University of Cambridge).
        - Watkins, C. J., & Dayan, P. (1992). Q-learning. Machine learning, 8(3-4), 279-292.
        - White, J. (2012). Bandit algorithms for website optimization. ” O’Reilly Media, Inc.”.
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

        Returns:
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
