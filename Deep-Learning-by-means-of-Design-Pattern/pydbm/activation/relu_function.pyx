# -*- coding: utf-8 -*-

import numpy as np

cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class ReLuFunction(ActivatingFunctionInterface):
    '''
    ReLu Function.
    '''

    # The length of memories.
    __memory_len = 50

    def __init__(self, memory_len=50):
        '''
        Init.

        Args:
            memory_len:     The number of memos of activities for derivative in backward.

        '''
        self.__mask_arr_list = []
        self.__memory_len = memory_len

    def activate(self, np.ndarray x):
        '''
        Activate and extract feature points in forward propagation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of the activated feature points.
        '''
        self.__mask_arr_list.append((x <= 0))
        if len(self.__mask_arr_list) > self.__memory_len:
            self.__mask_arr_list = self.__mask_arr_list[len(self.__mask_arr_list) - self.__memory_len:]

        x = np.maximum(0, x).astype(np.float64)
        if self.batch_norm is not None:
            x = self.batch_norm.forward_propagation(x)

        return x

    def derivative(self, np.ndarray y):
        '''
        Derivative and extract delta in back propagation.

        Args:
            y:  `np.ndarray` of delta.

        Returns:
            `np.ndarray` of delta.
        '''
        if self.batch_norm is not None:
            y = self.batch_norm.back_propagation(y)

        mask_arr = self.__mask_arr_list.pop(-1)
        y[mask_arr] = 0
        return y

    def forward(self, np.ndarray x):
        '''
        Forward propagation but not retain the activation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            The result.
        '''
        return np.maximum(0, x).astype(np.float64)

    def backward(self, np.ndarray y):
        '''
        Back propagation but not operate the activation.

        Args:
            y:  `np.ndarray` of delta.

        Returns:
            The result.
        '''
        mask_arr = self.__mask_arr_list[-1]
        y[mask_arr] = 0
        return y
