# -*- coding: utf-8 -*-

import math
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class TanhFunction(ActivatingFunctionInterface):
    '''
    Tanh function.
    '''

    def activate(self, double x):
        '''
        Return the result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        x_sum = x.sum()
        if x_sum != 0:
            x = x / x_sum
        return math.tanh(x)
