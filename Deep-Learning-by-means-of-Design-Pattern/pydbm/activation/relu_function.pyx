# -*- coding: utf-8 -*-

import numpy as np

cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class ReLuFunction(ActivatingFunctionInterface):
    '''
    ReLu Function.
    '''
    
    def __init__(self, normalize_flag=False):
        '''
        Init.
        
        Args:
            normalize_flag:     Z-Score normalize or not.

        '''
        self.__normalize_flag = normalize_flag

    def activate(self, np.ndarray x):
        '''
        Return of result from this activation function.

        Args:
            x:                  Parameter.

        Returns:
            The result.
        '''
        cdef double x_mean
        cdef double x_std

        if self.__normalize_flag is True:
            x_mean = x.mean()
            x_std = x.std()
            if x_std != 0:
                x = (x - x_mean) / x_std

        x = np.maximum(0, x).astype(np.float64)
        return x

    def derivative(self, np.ndarray y):
        '''
        Return of derivative result from this activation function.

        Args:
            y:   The result of activation.

        Returns:
            The result.
        '''
        return (y > 0).astype(int).astype(np.float64)
