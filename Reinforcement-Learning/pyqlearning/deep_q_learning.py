# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
from pyqlearning.function_approximator import FunctionApproximator


class DeepQLearning(metaclass=ABCMeta):
    '''
    Abstract base class to implement the Deep Q-Learning.
    
    The Reinforcement learning theory presents several issues 
    from a perspective of deep learning theory(Mnih, V., et al. 2013).
    Firstly, deep learning applications have required large amounts of 
    hand-labelled training data. Reinforcement learning algorithms, 
    on the other hand, must be able to learn from a scalar reward signal 
    that is frequently sparse, noisy and delayed.

    The difference between the two theories is not only the type of data 
    but also the timing to be observed. The delay between taking actions and 
    receiving rewards, which can be thousands of timesteps long, seems particularly 
    daunting when compared to the direct association between inputs and targets found 
    in supervised learning.

    Another issue is that deep learning algorithms assume the data samples to be independent, 
    while in reinforcement learning one typically encounters sequences of highly correlated 
    states. Furthermore, in Reinforcement learning, the data distribution changes as the 
    algorithm learns new behaviours, presenting aspects of recursive learning, which can be 
    problematic for deep learning methods that assume a fixed underlying distribution.

    Increasing the complexity of states/actions is equivalent to increasing the number of 
    combinations of states/actions. If the value function is continuous and granularities of 
    states/actions are extremely fine, the combinatorial explosion will be encountered. 
    In other words, this basic approach is totally impractical, because the state/action-value 
    function is estimated separately for each sequence, without any **generalisation**. Instead, 
    it is common to use a **function approximator** to estimate the state/action-value function.

    Considering many variable parts and functional extensions in the Deep Q-learning paradigm 
    from perspective of commonality/variability analysis in order to practice 
    object-oriented design, this abstract class defines the skeleton of a Deep Q-Learning 
    algorithm in an operation, deferring some steps in concrete variant algorithms 
    such as Epsilon Deep Q-Network to client subclasses. This abstract class in this library 
    lets subclasses redefine certain steps of a Deep Q-Learning algorithm without changing 
    the algorithm’s structure.

    References:
        - Egorov, M. (2016). Multi-agent deep reinforcement learning.(URL: https://pdfs.semanticscholar.org/dd98/9d94613f439c05725bad958929357e365084.pdf)
        - Gupta, J. K., Egorov, M., & Kochenderfer, M. (2017, May). Cooperative multi-agent control using deep reinforcement learning. In International Conference on Autonomous Agents and Multiagent Systems (pp. 66-83). Springer, Cham.
        - Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
    '''

    # Learning rate.
    __alpha_value = 0.1
    # Discount.
    __gamma_value = 0.5

    def __init__(self, function_approximator):
        '''
        Init.
        
        Args:
            function_approximator:  is-a `FunctionApproximator`.
        '''
        if isinstance(function_approximator, FunctionApproximator):
            self.__function_approximator = function_approximator
        else:
            raise TypeError()
        self.t = 1

    def learn(self, state_arr, limit=1000):
        '''
        Learning and searching the optimal solution.
        
        Args:
            state_arr:      `np.ndarray` of initial state.
            limit:          The maximum number of iterative updates based on value iteration algorithms.
        '''
        while self.t <= limit:
            next_action_arr = self.extract_possible_actions(state_arr)
            next_q_arr = self.__function_approximator.inference_q(next_action_arr)
            action_arr, q = self.select_action(next_action_arr, next_q_arr)
            reward_value = self.observe_reward_value(state_arr, action_arr)

            # Max-Q-Value in next action time.
            next_next_action_arr = self.extract_possible_actions(action_arr)
            next_max_q = self.__function_approximator.inference_q(next_next_action_arr).max()

            # Update Q-Value.
            self.update_q(
                q=q,
                reward_value=reward_value,
                next_max_q=next_max_q
            )

            # Update State.
            state_arr = self.update_state(state_arr, action_arr)

            # Epsode.
            self.t += 1
            # Check.
            end_flag = self.check_the_end_flag(state_arr)
            if end_flag is True:
                break

    @abstractmethod
    def extract_possible_actions(self, state_arr):
        '''
        Extract possible actions.

        Args:
            state_arr:  `np.ndarray` of state.
        
        Returns:
            `np.ndarray` of actions.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def select_action(self, next_action_arr, next_q_arr):
        '''
        Select action by Q(state, action).

        Args:
            next_action_arr:        `np.ndarray` of actions.
            next_q_arr:             `np.ndarray` of Q-Values.

        Returns:
            Tuple(`np.ndarray` of action., Q-Value)
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def observe_reward_value(self, state_arr, action_arr):
        '''
        Compute the reward value.
        
        Args:
            state_arr:              `np.ndarray` of state.
            action_arr:             `np.ndarray` of action.
        
        Returns:
            Reward value.
        '''
        raise NotImplementedError("This method must be implemented.")

    def update_q(self, q, reward_value, next_max_q):
        '''
        Update Q.
        
        Args:
            q:              Q-Value.
            reward_value:   Reward value.
            next_max_q:     Maximum Q-Value in next time step.
        '''
        # Update Q-Value.
        new_q = q + (self.alpha_value * (reward_value + (self.gamma_value * next_max_q) - q))
        # Learn updated Q-Value.
        self.__function_approximator.learn_q(q, new_q)

    def update_state(self, state_arr, action_arr):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.
            action_arr:   `np.ndarray` of action in `self.t`.
        
        Returns:
            `np.ndarray` of state in `self.t+1`.
        '''
        return action_arr

    def check_the_end_flag(self, state_arr):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:    `np.ndarray` of state in `self.t`.

        Returns:
            bool
        '''
        # As a rule, the learning can not be stopped.
        return False

    def get_function_approximator(self):
        ''' getter '''
        return self.__function_approximator

    def set_function_approximator(self, value):
        if isinstance(value, FunctionApproximator):
            self.__function_approximator = value
        else:
            raise TypeError()

    function_approximator = property(get_function_approximator, set_function_approximator)

    def get_alpha_value(self):
        '''
        getter
        Learning rate.
        '''
        if isinstance(self.__alpha_value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        return self.__alpha_value

    def set_alpha_value(self, value):
        '''
        setter
        Learning rate.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __alpha_value must be float.")
        self.__alpha_value = value

    alpha_value = property(get_alpha_value, set_alpha_value)

    def get_gamma_value(self):
        '''
        getter
        Gamma value.
        '''
        if isinstance(self.__gamma_value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        return self.__gamma_value

    def set_gamma_value(self, value):
        '''
        setter
        Gamma value.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of __gamma_value must be float.")
        self.__gamma_value = value

    gamma_value = property(get_gamma_value, set_gamma_value)
