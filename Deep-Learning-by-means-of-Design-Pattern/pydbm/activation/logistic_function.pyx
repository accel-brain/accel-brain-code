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
    
    def __init__(self, binary_flag=False, normalize_flag=True):
        if isinstance(binary_flag, bool):
            self.__binary_flag = binary_flag
        else:
            raise TypeError()
        
        if isinstance(normalize_flag, bool):
            self.__normalize_flag = normalize_flag
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
        cdef double x_sum
        if self.__normalize_flag is True:
            x_sum = x.sum()
            if x_sum != 0:
                x = x / x_sum

        if self.__binary_flag is True:
            activity_arr = 1.0 / (1.0 + np.exp(-x))
            activity_arr = np.random.binomial(1, activity_arr, activity_arr.shape[0])
            activity_arr = activity_arr.astype(np.float64)
            return activity_arr
        else:
            return 1.0 / (1.0 + np.exp(-x))
