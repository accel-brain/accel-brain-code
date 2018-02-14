# -*- coding: utf-8 -*-

import numpy as np

cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class ReLuFunction(ActivatingFunctionInterface):
    '''
    ReLu Function.
    '''

    def activate(self, x):
        '''
        Return of result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        x_sum = x.sum()
        if x_sum != 0:
            x = x / x_sum
        return np.maximum(0, x)
