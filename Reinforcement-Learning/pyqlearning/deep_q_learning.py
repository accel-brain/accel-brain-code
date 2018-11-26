# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
from pyqlearning.function_approximator import FunctionApproximator


class DeepQLearning(metaclass=ABCMeta):
    '''
    Abstract base class to implement the Deep Q-Learning.
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
        Learning.
        
        Args:
            state_arr:      `np.ndarray` of initial state.
            limit:          The number of learning.
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
            The shape is:(
                `batch size corresponded to each action key`, 
                `channel that is 1`, 
                `feature points1`, 
                `feature points2`
            )
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
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
