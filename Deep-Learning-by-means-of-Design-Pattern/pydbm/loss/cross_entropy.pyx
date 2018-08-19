# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss


class CrossEntropy(ComputableLoss):
    '''
    Cross Entropy.
    '''

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

        if labeled_arr.size == pred_arr.size:
            _labeled_arr = labeled_arr.argmax(axis=1)
        else:
            _labeled_arr = labeled_arr

        cdef int batch_size = pred_arr.shape[0]
        cdef np.ndarray _pred_arr = pred_arr[np.arange(batch_size), _labeled_arr]
        _pred_arr = ((1 - 1e-15) * (_pred_arr - _pred_arr.min()) / (_pred_arr.max() - _pred_arr.min())) + 1e-15
        return -np.sum(np.log(_pred_arr), axis=axis) / batch_size

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

        if labeled_arr.size == pred_arr.size:
            _labeled_arr = labeled_arr.argmax(axis=1)
        else:
            _labeled_arr = labeled_arr

        batch_size = pred_arr.shape[0]
        pred_arr[np.arange(batch_size), _labeled_arr] -= 1
        pred_arr *= delta_output
        pred_arr = pred_arr / batch_size
        return pred_arr
