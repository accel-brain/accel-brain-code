# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss


class MeanSquaredError(ComputableLoss):
    '''
    The mean squared error (MSE).
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
        return np.square(labeled_arr - pred_arr).mean(axis=axis)
