# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss


class MeanSquaredError(ComputableLoss):
    '''
    The mean squared error (MSE).
    '''

    def compute_loss(self, np.ndarray pred_arr, np.ndarray labeled_arr):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.

        Returns:
            Cost.
        '''
        return np.square(labeled_arr - pred_arr).mean()

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
        return (pred_arr - labeled_arr) / batch_size * delta_output
