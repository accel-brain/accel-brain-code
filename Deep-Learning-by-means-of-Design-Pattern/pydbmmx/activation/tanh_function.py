# -*- coding: utf-8 -*-
import mxnet as mx
from pydbmmx.activation.interface.activating_function_interface import ActivatingFunctionInterface


class TanhFunction(ActivatingFunctionInterface):
    '''
    Tanh function.
    '''

    def activate(self, x):
        '''
        Return the result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        return mx.ndarray.tanh(x)

    def derivative(self, y):
        '''
        Derivative.

        Args:
            y:  Paramter.
        Returns:
            The result.
        '''
        return 1.0 - y**2
