# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class ComputableLoss(metaclass=ABCMeta):
    '''
    Interface of Loss functions.
    '''

    @abstractmethod
    def compute_loss(self, np.ndarray pred_arr, np.ndarray labeled_arr):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.

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
