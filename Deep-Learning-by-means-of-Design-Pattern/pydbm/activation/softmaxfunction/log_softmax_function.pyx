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
        return x - np.log(np.nansum(np.exp(x - np.max(x)), axis=1).reshape(-1, 1)) + np.max(x)
