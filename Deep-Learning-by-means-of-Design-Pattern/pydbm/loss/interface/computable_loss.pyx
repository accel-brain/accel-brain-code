# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class ComputableLoss(metaclass=ABCMeta):
    '''
    Interface of Loss functions.
    '''

    # Penalty term.
    __penalty_term = 0.0

    # Penalty delta.
    __penalty_delta_arr = None

    @abstractmethod
    def compute_loss(self, np.ndarray pred_arr, np.ndarray labeled_arr, axis=None):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            axis:           Axis or axes along which the losses are computed.
                            The default is to compute the losses of the flattened array.

        Returns:
            Cost.
        '''
        raise NotImplementedError()

    @abstractmethod
    def compute_delta(self, np.ndarray pred_arr, np.ndarray labeled_arr, delta_output=1):
        '''
        Backward delta.
        
        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            delta_output:   Delta.

        Returns:
            Delta.
        '''
        raise NotImplementedError()

    def get_penalty_term(self):
        ''' getter '''
        return self.__penalty_term

    def set_penalty_term(self, value):
        ''' setter '''
        self.__penalty_term = value

    penalty_term = property(get_penalty_term, set_penalty_term)

    def get_penalty_delta_arr(self):
        ''' getter '''
        return self.__penalty_delta_arr
    
    def set_penalty_delta_arr(self, value):
        ''' setter '''
        self.__penalty_delta_arr = value
    
    penalty_delta_arr = property(get_penalty_delta_arr, set_penalty_delta_arr)
