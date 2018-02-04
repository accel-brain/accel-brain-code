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
        return math.tanh(x)

    def derivative(self, double y):
        '''
        Derivative.

        Args:
            y:  Paramter.
        Returns:
            The result.
        '''
        return 1.0 - y**2
