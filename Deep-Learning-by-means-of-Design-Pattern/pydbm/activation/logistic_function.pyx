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
    __overflow_range = 710.0
    
    # Normalization mode.
    __normalization_mode = "sum_partition"

    def __init__(
        self,
        binary_flag=False,
        normalize_flag=False,
        for_overflow="max",
        normalization_mode="sum_partition"
    ):
        '''
        Init.
        
        Args:
            binary_flag:        Input binary data(0-1) or not.
            normalize_flag:     Normalize or not, before the activation.
            for_overflow:       If this value is `max`, the activation function is as follows:
                                $\frac{1.0}{(1.0 + \exp(-x + c))}$
                                where $c$ is maximum value of `x`.

            normalization_mode: How to normalize `x`.
                                `sum_partition`: $x = \frac{x}{\sum_{}^{}x}$
                                `z_score`: $x = \frac{(x - \mu)}{\sigma}$
                                           where $\mu$ is mean of `x` and $\sigma$ is standard deviation of `x`.
        '''
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
        cdef double x_sum
        cdef double x_std

        if self.__normalize_flag is True:
            if self.__normalization_mode == "sum_partition":
                x_sum = x.sum()
                if x_sum != 0.0:
                    x = x / x_sum
            elif self.__normalization_mode == "z_score":
                x_std = x.std()
                if x_std != 0.0:
                    x = (x - x.mean()) / x_std

        if self.__for_overflow == "max":
            c = x.max()
        else:
            c = 0.0

        cdef np.ndarray c_arr = -x + c
        c_arr[c_arr >= self.__overflow_range] = self.__overflow_range

        activity_arr = 1.0 / (1.0 + np.exp(c_arr))
        activity_arr = np.nan_to_num(activity_arr)

        if self.__for_overflow == "max":
            if activity_arr.max() - activity_arr.min() > 0:
                activity_arr = (activity_arr - activity_arr.min()) / (activity_arr.max() - activity_arr.min())

        if self.__binary_flag is True:
            activity_arr = np.random.binomial(1, activity_arr, activity_arr.shape[0])
            activity_arr = activity_arr.astype(np.float64)

        return activity_arr
