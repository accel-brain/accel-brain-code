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
    # The logs of predicted and real Q-Values.
    __q_logs_arr = np.array([])

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
        self.__q_logs_arr = np.array([])

    def learn(self, state_arr, limit=1000):
        '''
        Learning and searching the optimal solution.
        
        Args:
            state_arr:      `np.ndarray` of initial state.
            limit:          The maximum number of iterative updates based on value iteration algorithms.
        '''
        while self.t <= limit:
            # Draw samples of next possible actions from any distribution.
            next_action_arr = self.extract_possible_actions(state_arr)
            # Inference Q-Values.
            predicted_q_arr = self.__function_approximator.inference_q(next_action_arr)
            # Set `np.ndarray` of rewards and next Q-Values.
            reward_value_arr = np.empty((next_action_arr.shape[0], 1))
            next_max_q_arr = np.empty((next_action_arr.shape[0], 1))
            for i in range(reward_value_arr.shape[0]):
                # Observe reward values.
                reward_value_arr[i] = self.observe_reward_value(state_arr, next_action_arr[i])
                # Inference the Max-Q-Value in next action time.
                next_next_action_arr = self.extract_possible_actions(next_action_arr[i])
                next_max_q_arr[i] = self.__function_approximator.inference_q(next_next_action_arr).max()

            # Select action.
            action_arr, predicted_q = self.select_action(next_action_arr, predicted_q_arr)
            # Update real Q-Values.
            real_q_arr = self.update_q(
                predicted_q_arr,
                reward_value_arr,
                next_max_q_arr
            )

            # Maximum of predicted and real Q-Values.
            real_q = real_q_arr[np.where(predicted_q_arr == predicted_q)[0][0]]
            if self.__q_logs_arr.shape[0] > 0:
                self.__q_logs_arr = np.r_[
                    self.__q_logs_arr,
                    np.array([predicted_q, real_q]).reshape(1, 2)
                ]
            else:
                self.__q_logs_arr = np.array([predicted_q, real_q]).reshape(1, 2)

            # Learn Q-Values.
            self.learn_q(predicted_q_arr, real_q_arr)
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

    def update_q(self, predicted_q_arr, reward_value_arr, next_max_q_arr):
        '''
        Update Q.
        
        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            reward_value_arr:   `np.ndarray` of reward values.
            next_max_q_arr:     `np.ndarray` of maximum Q-Values in next time step.
        
        Returns:
            `np.ndarray` of real Q-Values.
        '''
        # Update Q-Value.
        return predicted_q_arr + (self.alpha_value * (reward_value_arr + (self.gamma_value * next_max_q_arr) - predicted_q_arr))

    def learn_q(self, predicted_q_arr, real_q_arr):
        '''
        Learn Q with the function approximator.

        Args:
            predicted_q_arr:    `np.ndarray` of predicted Q-Values.
            real_q_arr:         `np.ndarray` of real Q-Values.
        '''
        # Learn updated Q-Value.
        self.__function_approximator.learn_q(predicted_q_arr, real_q_arr)

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

    def get_q_logs_arr(self):
        ''' getter '''
        return self.__q_logs_arr
    
    def set_q_logs_arr(self, values):
        ''' setter '''
        raise TypeError("The `q_logs_arr` must be read-only.")
    
    q_logs_arr = property(get_q_logs_arr, set_q_logs_arr)
