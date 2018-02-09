# -*- coding: utf-8 -*-
import mxnet as mx
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


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
        return mx.ndarray.maximum(0, x)

    def derivative(self, y):
        if y < 0:
            return 0.0
        elif y > 0:
            return 1.0
        else:
            raise ValueError("The derivative does not exist.")
