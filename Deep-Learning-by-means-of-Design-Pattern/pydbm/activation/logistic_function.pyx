# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class LogisticFunction(ActivatingFunctionInterface):
    '''
    Logistic Function.
    '''
    
    # Normalize flag.
    __normalize_flag = False

    # Binary flag.
    __binary_flag = False
    
    # for overflow
    __for_overflow = "max"
    
    # Range of x.
    __overflow_range = 34.538776394910684

    def __init__(self, binary_flag=False, normalize_flag=False, for_overflow="max"):
        if isinstance(binary_flag, bool):
            self.__binary_flag = binary_flag
        else:
            raise TypeError()
        
        if isinstance(normalize_flag, bool):
            self.__normalize_flag = normalize_flag
        else:
            raise TypeError()
        
        if isinstance(for_overflow, str):
            self.__for_overflow = for_overflow
        else:
            raise TypeError()

    def activate(self, np.ndarray x):
        '''
        Return of result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        x[x <= -self.__overflow_range] = 1e-15
        x[x >= self.__overflow_range] = 1.0 - 1e-15

        cdef double x_sum
        if self.__normalize_flag is True:
            x_sum = x.sum()
            if x_sum != 0:
                x = x / x_sum

        if self.__for_overflow == "max":
            c = x.max()
        else:
            c = 0.0

        activity_arr = 1.0 / (1.0 + np.exp(-x + c))
        activity_arr = np.nan_to_num(activity_arr)

        if self.__binary_flag is True:
            activity_arr = np.random.binomial(1, activity_arr, activity_arr.shape[0])
            activity_arr = activity_arr.astype(np.float64)

        return activity_arr

    def derivative(self, np.ndarray x):
        '''
        Return of derivative result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        return x * (1 - x)
