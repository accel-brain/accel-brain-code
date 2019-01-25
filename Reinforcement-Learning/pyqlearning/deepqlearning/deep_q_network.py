# -*- coding: utf-8 -*-
import numpy as np
from pyqlearning.deep_q_learning import DeepQLearning


class DeepQNetwork(DeepQLearning):
    '''
    Abstract base class to implement the Deep Q-Network(DQN).

    The structure of Q-Learning is based on the Epsilon Greedy Q-Leanring algorithm,
    which is a typical off-policy algorithm.  In this paradigm, stochastic searching 
    and deterministic searching can coexist by hyperparameter `epsilon_greedy_rate` 
    that is probability that agent searches greedy. Greedy searching is deterministic 
    in the sensethat policy of agent follows the selection that maximizes the Q-Value.

    References:
        - https://code.accel-brain.com/Reinforcement-Learning/README.html#deep-q-network
        - Egorov, M. (2016). Multi-agent deep reinforcement learning.(URL: https://pdfs.semanticscholar.org/dd98/9d94613f439c05725bad958929357e365084.pdf)
        - Gupta, J. K., Egorov, M., & Kochenderfer, M. (2017, May). Cooperative multi-agent control using deep reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (pp. 66-83). Springer, Cham.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
    '''

    # ε-greedy rate.
    __epsilon_greedy_rate = 0.75

    def select_action(self, next_action_arr, next_q_arr):
        '''
        Select action by Q(state, action).

        Args:
            next_action_arr:        `np.ndarray` of actions.
            next_q_arr:             `np.ndarray` of Q-Values.

        Retruns:
            Tuple(`np.ndarray` of action., Q-Value)
        '''
        key_arr = self.select_action_key(next_action_arr, next_q_arr)
        return next_action_arr[key_arr], next_q_arr[key_arr]

    def select_action_key(self, next_action_arr, next_q_arr):
        '''
        Select action by Q(state, action).

        Args:
            next_action_arr:        `np.ndarray` of actions.
            next_q_arr:             `np.ndarray` of Q-Values.

        Retruns:
            `np.ndarray` of keys.
        '''
        epsilon_greedy_flag = bool(np.random.binomial(n=1, p=self.epsilon_greedy_rate))
        if epsilon_greedy_flag is False:
            key = np.random.randint(low=0, high=next_action_arr.shape[0])
        else:
            key = next_q_arr.argmax()

        return key

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
