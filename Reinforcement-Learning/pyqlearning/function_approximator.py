# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class FunctionApproximator(metaclass=ABCMeta):
    '''
    The interface of Function Approximators.
    '''

    @abstractmethod
    def learn_q(self, q, new_q):
        '''
        Infernce Q-Value.
        
        Args:
            q:                  Predicted Q-Value.
            new_q:              Real Q-Value.
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def inference_q(self, next_action_arr):
        '''
        Infernce Q-Value.
        
        Args:
            next_action_arr:     `np.ndarray` of action.
        
        Returns:
            `np.ndarray` of Q-Values.
        '''
        raise NotImplementedError("This method must be implemented.")
