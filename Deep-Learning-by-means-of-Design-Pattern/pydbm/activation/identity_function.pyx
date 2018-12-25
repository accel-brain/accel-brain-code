# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class IdentityFunction(ActivatingFunctionInterface):
    '''
    Identity function.
    '''

    def activate(self, np.ndarray x):
        '''
        Return the result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        return x

    def derivative(self, np.ndarray y):
        '''
        Return of derivative with respect to this activation function.

        Args:
            y   The result of activation.

        Returns:
            The result.
        '''
        return y
