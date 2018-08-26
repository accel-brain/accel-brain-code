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
    __overflow_range = 708.0
    
    # Normalization mode.
    __normalization_mode = "sum_partition"

    def __init__(
        self,
        binary_flag=False,
        normalize_flag=True,
        for_overflow="max",
        normalization_mode="min_max"
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
                                `min_max`: $x = \frac{x - x_{min}}{x_{max} - x_{min}}$
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
        cdef double partition

        if self.__normalize_flag is True:
            try:
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
            except FloatingPointError:
                pass

        if self.__for_overflow == "max":
            c = x.max()
        else:
            c = 0.0

        cdef np.ndarray c_arr = np.nansum(
            np.array([
                np.expand_dims(-x, axis=0), 
                np.expand_dims(np.ones_like(x) * c, axis=0)
            ]),
            axis=0
        )[0]
        if c_arr[c_arr >= self.__overflow_range].shape[0] > 0 or c_arr[c_arr < -self.__overflow_range].shape[0] > 0:
            c_max = c_arr.max()
            c_min = c_arr.min()
            if c_max != c_min:
                c_arr = np.nansum(
                    np.array([
                        np.expand_dims(c_arr, axis=0),
                        np.expand_dims(np.ones_like(c_arr) * c_min * -1, axis=0)
                    ]),
                    axis=0
                )[0]
                partition = np.nansum(np.array([c_max, -1 * c_min]))
                c_arr = np.nanprod(
                    np.array([
                        np.expand_dims(c_arr, axis=0),
                        np.expand_dims(np.ones_like(c_arr) / partition, axis=0)
                    ]),
                    axis=0
                )[0]

                c_arr = np.nanprod(
                    np.array([
                        np.expand_dims(c_arr, axis=0),
                        np.expand_dims(
                            np.ones_like(c_arr) * (self.__overflow_range - (-self.__overflow_range)),
                            axis=0
                        )
                    ]),
                    axis=0
                )[0]
                c_arr = np.nansum(
                    np.array([
                        np.expand_dims(c_arr, axis=0),
                        np.expand_dims(np.ones_like(c_arr) * -self.__overflow_range, axis=0)
                    ]),
                    axis=0
                )[0]

        activity_arr = 1.0 / (1.0 + np.exp(c_arr))
        activity_arr = np.nan_to_num(activity_arr)

        if self.__for_overflow == "max":
            x_max = activity_arr.max()
            x_min = activity_arr.min()
            if x_max != x_min:
                activity_arr = np.nansum(
                    np.array([
                        np.expand_dims(activity_arr, axis=0),
                        np.expand_dims(np.ones_like(activity_arr) * x_min * -1, axis=0)
                    ]),
                    axis=0
                )[0]
                partition = np.nansum(np.array([x_max, -1 * x_min]))
                activity_arr = np.nanprod(
                    np.array([
                        np.expand_dims(activity_arr, axis=0),
                        np.expand_dims(np.ones_like(activity_arr) / partition, axis=0)
                    ]),
                    axis=0
                )[0]

        if self.__binary_flag is True:
            activity_arr = np.random.binomial(1, activity_arr, activity_arr.shape)
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
