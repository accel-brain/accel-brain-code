# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.activation.softmax_function import SoftmaxFunction
ctypedef np.float64_t DOUBLE_t


class LogSoftmaxFunction(SoftmaxFunction):
    '''
    log-Softmax function.
    '''

    def forward(self, np.ndarray x):
        '''
        Forward propagation but not retain the activation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            The result.
        '''
        max_x = np.max(x)
        cdef np.ndarray exp_x_arr = np.exp(x - max_x)
        cdef np.ndarray sum_x_arr = np.nansum(exp_x_arr, axis=1).reshape(-1, 1)
        sum_x_arr[sum_x_arr == 0.0] += 1e-08
        return np.log((x - max_x) / sum_x_arr)
