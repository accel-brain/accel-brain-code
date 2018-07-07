# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class OptimizableLoss(metaclass=ABCMeta):
    '''
    Interface of Loss functions.
    '''

    @abstractmethod
    def compute_loss(self, np.ndarray pred_arr, np.ndarray labeled_arr, axis=None):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            axis:           None or int or tuple of ints, optional.
                            Axis or axes along which the means are computed.
                            The default is to compute the mean of the flattened array.

        Returns:
            Cost.
        '''
        raise NotImplementedError()
