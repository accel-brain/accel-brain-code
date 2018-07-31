# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss


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
            axis:           None or int or tuple of ints, optional.
                            Axis or axes along which the means are computed.
                            The default is to compute the mean of the flattened array.

        Returns:
            Cost.
        '''
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(1, pred_arr.size)
            labeled_arr = labeled_arr.reshape(1, labeled_arr.size)
        batch_size = pred_arr.shape[0]
        return -np.sum(np.log(pred_arr[np.arange(batch_size), labeled_arr] + 1e-15), axis=axis) / batch_size

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

        batch_size = labeled_arr.shape[0]
        pred_arr[np.arange(batch_size), labeled_arr] -= 1
        pred_arr *= delta_output
        pred_arr = pred_arr / batch_size
        return pred_arr
