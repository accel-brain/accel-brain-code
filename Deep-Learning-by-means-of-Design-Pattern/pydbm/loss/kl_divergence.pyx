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
        pred_arr = (pred_arr - pred_arr.min()) / (pred_arr.max() - pred_arr.min() + 1e-08)
        labeled_arr = (labeled_arr - labeled_arr.min()) / (labeled_arr.max() - labeled_arr.min() + 1e-08)

        labeled_arr[labeled_arr == 0] += 1e-08
        cdef np.ndarray diff_arr = pred_arr * np.ma.log((pred_arr / labeled_arr))

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
        labeled_arr[labeled_arr == 0] += 1e-08
        cdef np.ndarray delta_arr = pred_arr * np.ma.log(pred_arr / labeled_arr)
        delta_arr = delta_arr * delta_output
        v = np.linalg.norm(delta_arr)
        if v > self.__grad_clip_threshold:
            delta_arr = delta_arr * self.__grad_clip_threshold / v

        if self.penalty_delta_arr is not None:
            delta_arr += self.penalty_delta_arr

        self.penalty_delta_arr = None
        return delta_arr
