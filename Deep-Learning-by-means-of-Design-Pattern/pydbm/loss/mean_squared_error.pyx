# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss


class MeanSquaredError(ComputableLoss):
    '''
    The mean squared error (MSE).

    References:
        - Pascanu, R., Mikolov, T., & Bengio, Y. (2012). Understanding the exploding gradient problem. CoRR, abs/1211.5063, 2.
        - Pascanu, R., Mikolov, T., & Bengio, Y. (2013, February). On the difficulty of training recurrent neural networks. In International conference on machine learning (pp. 1310-1318).
    '''

    def __init__(self, grad_clip_threshold=1e+05):
        '''
        Init.

        Args:
            grad_clip_threshold:    Threshold of the gradient clipping.
        '''
        self.penalty_term = 0.0
        self.penalty_delta_arr = None
        self.__grad_clip_threshold = grad_clip_threshold

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
        cdef np.ndarray diff_arr = (labeled_arr - pred_arr)
        v = np.linalg.norm(diff_arr)
        if v > self.__grad_clip_threshold:
            diff_arr = diff_arr * self.__grad_clip_threshold / v

        loss = np.square(diff_arr).mean(axis=axis) + self.penalty_term
        self.penalty_term = 0.0
        return loss

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
        cdef np.ndarray delta_arr = (pred_arr - labeled_arr) * delta_output
        v = np.linalg.norm(delta_arr)
        if v > self.__grad_clip_threshold:
            delta_arr = delta_arr * self.__grad_clip_threshold / v

        if self.penalty_delta_arr is not None:
            delta_arr += self.penalty_delta_arr

        self.penalty_delta_arr = None
        return delta_arr
