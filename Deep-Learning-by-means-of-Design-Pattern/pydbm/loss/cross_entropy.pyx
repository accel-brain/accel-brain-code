# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss


class CrossEntropy(ComputableLoss):
    '''
    Cross Entropy.
    '''

    def __init__(self):
        '''
        Init.
        '''
        self.penalty_term = 0.0
        self.penalty_delta_arr = None

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
        cdef np.ndarray _labeled_arr

        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(1, pred_arr.size)
            labeled_arr = labeled_arr.reshape(1, labeled_arr.size)

        if pred_arr.ndim > 2:
            pred_arr = pred_arr.reshape((pred_arr.shape[0], -1))
        if labeled_arr.ndim > 2:
            labeled_arr = labeled_arr.reshape((labeled_arr.shape[0], -1))

        if labeled_arr.size == pred_arr.size or labeled_arr.ndim == 2:
            _labeled_arr = labeled_arr.argmax(axis=1)
        else:
            _labeled_arr = labeled_arr

        cdef int batch_size = pred_arr.shape[0]
        cdef np.ndarray _pred_arr = pred_arr[np.arange(batch_size), _labeled_arr]

        loss = -np.sum(np.ma.log(_pred_arr), axis=axis) / batch_size
        loss = loss + self.penalty_term
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
        cdef np.ndarray _labeled_arr
        cdef np.ndarray delta_arr
        cdef int batch_size = pred_arr.shape[0]

        if pred_arr.ndim > 2:
            delta_arr = pred_arr.reshape((pred_arr.shape[0], -1))
        else:
            delta_arr = pred_arr

        if labeled_arr.ndim > 2:
            labeled_arr = labeled_arr.reshape((labeled_arr.shape[0], -1))

        if labeled_arr.size == delta_arr.size or labeled_arr.ndim == 2:
            _labeled_arr = labeled_arr.argmax(axis=1)
        else:
            _labeled_arr = labeled_arr

        delta_arr[np.arange(batch_size), _labeled_arr] -= 1
        delta_arr *= delta_output
        delta_arr = delta_arr / batch_size

        if pred_arr.ndim != delta_arr.ndim:
            delta_arr = delta_arr.reshape(*pred_arr.copy().shape)

        if self.penalty_delta_arr is not None:
            if delta_arr.ndim != self.penalty_delta_arr.ndim:
                penalty_delta_arr = self.penalty_delta_arr.reshape(*delta_arr.copy().shape)
            else:
                penalty_delta_arr = self.penalty_delta_arr
            delta_arr += penalty_delta_arr

        self.penalty_delta_arr = None
        return delta_arr
