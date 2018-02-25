# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class SoftmaxFunction(ActivatingFunctionInterface):
    '''
    Softmax function.
    '''

    def activate(self, np.ndarray[DOUBLE_t, ndim=1] x):
        '''
        Return the result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        exp_x = np.exp(x - np.max(x))
        prob = exp_x / exp_x.sum(axis=0)
        return prob
