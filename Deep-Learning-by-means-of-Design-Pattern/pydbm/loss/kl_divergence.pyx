# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss


class KLDivergence(ComputableLoss):
    '''
    Kullback–Leibler Divergence (KLD).
    '''

    def __init__(self, grad_clip_threshold=1e+05):
        '''
        Init.

        Args:
            grad_clip_threshold:    Threshold of the gradient clipping.
        '''
        self.penalty_arr = None
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
        cdef np.ndarray diff_arr = pred_arr * np.ma.log((pred_arr / labeled_arr))

        v = np.linalg.norm(diff_arr)
        if v > self.__grad_clip_threshold:
            diff_arr = diff_arr * self.__grad_clip_threshold / v

        if self.penalty_arr is not None:
            diff_arr += self.penalty_arr

        diff_arr += self.loss_only_penalty
        return diff_arr.mean(axis=axis)

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
        cdef np.ndarray delta_arr = pred_arr * np.ma.log(pred_arr / labeled_arr)
        delta_arr = delta_arr * delta_output
        v = np.linalg.norm(delta_arr)
        if v > self.__grad_clip_threshold:
            delta_arr = delta_arr * self.__grad_clip_threshold / v

        if self.penalty_arr is not None:
            delta_arr += self.penalty_arr

        return delta_arr
