# -*- coding: utf-8 -*-
import mxnet as mx
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


class ReLuFunction(ActivatingFunctionInterface):
    '''
    Logistic Function.
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
        pass