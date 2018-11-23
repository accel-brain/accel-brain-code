# -*- coding: utf-8 -*-
import numpy as np
from q_learning import QLearning
from function_approximator import FunctionApproximator


class DeepQNetwork(object):
    '''
    Deep Q-Network(DQN).
    '''
    
    def __init__(self, q_learning, function_approximator):
        '''
        Init.
        
        Args:
            q_learning:             is-a `QLearning`.
            function_approximator:  is-a `FunctionApproximator`.
        '''
        if isinstance(q_learning, QLearning):
            self.__q_learning = q_learning
        else:
            raise TypeError()

        if isinstance(function_approximator, FunctionApproximator):
            self.__function_approximator = function_approximator
        else:
            raise TypeError()

    def learn(self, state_key, limit=1000):
        '''
        Learning.
        
        Args:
            state_key:      Initial state.
            limit:          The number of learning.
        '''
        self.t = 1
        while self.t <= limit:
            self.__q_learning.t = self.t
            next_action_list = self.extract_possible_actions(state_key)
            action_key = self.__q_learning.select_action(
                state_key=state_key,
                next_action_list=next_action_list
            )
            reward_value = self.__q_learning.observe_reward_value(state_key, action_key)

            # Check.
            if self.__q_learning.check_the_end_flag(state_key) is True:
                end_flag = True

            # Max-Q-Value in next action time.
            next_next_action_list = self.extract_possible_actions(action_key)
            next_action_key = self.predict_next_action(action_key, next_next_action_list)
            next_max_q = self.__function_approximator.inference_q(action_key, next_action_key)

            # Update Q-Value.
            self.update_q(
                state_key=state_key,
                action_key=action_key,
                reward_value=reward_value,
                next_max_q=next_max_q
            )

            # Update State.
            state_key = self.__q_learning.update_state(state_key=state_key, action_key=action_key)

            # Epsode.
            self.t += 1
            self.__q_learning.t = self.t
            if end_flag is True:
                break

    def extract_possible_actions(self, state_key_arr):
        '''
        Extract possible actions.

        Args:
            state_key_arr:  `np.ndarray` of state.
        
        Returns:
            `np.ndarray` of actions.
            The shape is:(
                `batch size corresponded to each action key`, 
                `channel that is 1`, 
                `feature points1`, 
                `feature points2`
            )
        '''
        pass

    def predict_next_action(self, state_key_arr, next_action_arr):
        '''
        Predict next action by Q-Learning.

        Args:
            state_key_arr:      `np.ndarray` of state in `self.t+1`.
            next_action_list:   `np.ndarray` of the possible action in `self.t+1`.

        Returns:
            `np.ndarray` of action.
        '''
        pass

    def update_q(self, state_key_arr, action_key_arr, reward_value, next_max_q):
        '''
        Update Q.
        
        Args:
            state_key_arr:      `np.ndarray` of state.
            action_key_arr:     `np.ndarray` of action.
            reward_value:       Reward value.
            next_max_q:         Maximum Q-Value in next time step.
        '''
        # Inference Q-Value.
        q = self.__function_approximator.inference_q(state_key_arr, action_key_arr)
        # Update Q-Value.
        new_q = q + self.__q_learning.alpha_value * (reward_value + (self.__q_learning.gamma_value * next_max_q) - q)
        # Learn updated Q-Value.
        self.__function_approximator.learn_q(state_key_arr, action_key_arr, new_q)
