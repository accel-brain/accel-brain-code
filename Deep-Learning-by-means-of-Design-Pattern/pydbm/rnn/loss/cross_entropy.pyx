# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss


class CrossEntropy(ComputableLoss):
    '''
    Cross-Entropy as loss.
    '''

    # Range of x.
    __overflow_range = 34.538776394910684
    
    def __init__(self, clip_by_value=(1e-05, 1.0)):
        '''
        Init.
        
        Args:
            clip_by_value:      The tuple of parameters of min-max normalization: (Min value, Max value).
        
        '''
        self.__clip_by_value = clip_by_value

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
        cdef int N
        if axis is None or axis == 0:
            N = pred_arr.shape[0]
        else:
            N = pred_arr.shape[axis]

        labeled_arr[labeled_arr <= -self.__overflow_range] = 1e-15
        labeled_arr[labeled_arr >= self.__overflow_range] = 1.0 - 1e-15

        pred_arr[pred_arr <= -self.__overflow_range] = 1e-15
        pred_arr[pred_arr >= self.__overflow_range] = 1.0 - 1e-15

        if self.__clip_by_value is not None:
            min_v, max_v = self.__clip_by_value
            r_labeled_arr = 1 - labeled_arr
            r_pred_arr = 1 - pred_arr
            labeled_arr = min_v + ((max_v - min_v) * (labeled_arr - labeled_arr.min()) / (labeled_arr.max() - labeled_arr.min()))
            pred_arr = min_v + ((max_v - min_v) * (pred_arr - pred_arr.min()) / (pred_arr.max() - pred_arr.min()))

            r_labeled_arr = min_v + ((max_v - min_v) * (r_labeled_arr - r_labeled_arr.min()) / (r_labeled_arr.max() - r_labeled_arr.min()))
            r_pred_arr = min_v + ((max_v - min_v) * (r_pred_arr - r_pred_arr.min()) / (r_pred_arr.max() - r_pred_arr.min()))

        return -(1.0/N) * np.sum(
            np.multiply(labeled_arr, np.log(pred_arr)) + np.multiply((r_labeled_arr), np.log(r_pred_arr)),
            axis=axis
        )
