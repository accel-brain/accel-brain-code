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
    __overflow_range = 709.0
    
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
        cdef double x_max
        cdef double x_min
        cdef double c_max
        cdef double c_min

        if self.__normalize_flag is True:
            if self.__normalization_mode == "sum_partition":
                x_sum = x.sum()
                if x_sum != 0.0:
                    x = x / x_sum
            elif self.__normalization_mode == "z_score":
                x_std = x.std()
                if x_std != 0.0:
                    x = (x - x.mean()) / x_std
            elif self.__normalization_mode == "min_max":
                x_max = x.max()
                x_min = x.min()
                if x_max != x_min:
                    x = (x - x.min()) / (x.max() - x.min())

        if self.__for_overflow == "max":
            c = x.max()
        else:
            c = 0.0

        cdef np.ndarray c_arr = -x + c
        if c_arr[c_arr >= self.__overflow_range].shape[0] > 0:
            c_max = c_arr.max()
            c_min = c_arr.min()
            if c_max != c_min:
                c_arr = self.__overflow_range * (c_arr - c_min) / (c_max - c_min)

        activity_arr = 1.0 / (1.0 + np.exp(c_arr))
        activity_arr = np.nan_to_num(activity_arr)

        if self.__for_overflow == "max":
            x_max = activity_arr.max()
            x_min = activity_arr.min()
            if x_max != x_min:
                activity_arr = (activity_arr - x_min) / (x_max - x_min)

        if self.__binary_flag is True:
            activity_arr = np.random.binomial(1, activity_arr, activity_arr.shape[0])
            activity_arr = activity_arr.astype(np.float64)

        return activity_arr

    def derivative(self, np.ndarray y):
        '''
        Return of derivative result from this activation function.

        Args:
            y:   The result of activation.

        Returns:
            The result.
        '''
        return y * (1 - y)
