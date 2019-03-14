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
        Return of result from this activation function.

        Args:
            x:                  Parameter.

        Returns:
            The result.
        '''
        self.__mask_arr_list.append((x <= 0))
        if len(self.__mask_arr_list) > self.__memory_len:
            self.__mask_arr_list = self.__mask_arr_list[len(self.__mask_arr_list) - self.__memory_len:]

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
        mask_arr = self.__mask_arr_list.pop(-1)
        y[mask_arr] = 0
        return y
