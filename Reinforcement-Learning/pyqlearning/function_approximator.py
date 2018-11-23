# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class FunctionApproximator(metaclass=ABCMeta):
    '''
    The interface of Function Approximators.
    '''

    @abstractmethod
    def learn_q(self, state_key_arr, action_key_arr, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            state_key_arr:      `np.ndarray` of state.
            action_key_arr:     `np.ndarray` of action.
            new_q:              Q-Value.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def inference_q(self, state_key_arr, action_key_arr):
        '''
        Infernce Q-Value.
        
        Args:
            state_key_arr:      `np.ndarray` of state.
            action_key_arr:     `np.ndarray` of action.
        
        Returns:
            Q-Value.
        '''
        raise NotImplementedError("This method must be implemented.")
