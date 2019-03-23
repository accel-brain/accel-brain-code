# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
ctypedef np.float64_t DOUBLE_t


class TanhFunction(ActivatingFunctionInterface):
    '''
    Tanh function.
    '''

    # The length of memories.
    __memory_len = 50

    def __init__(self, memory_len=50):
        '''
        Init.

        Args:
            memory_len:     The number of memos of activities for derivative in backward.

        '''
        self.__activity_arr_list = []
        self.__memory_len = memory_len

    def activate(self, np.ndarray x):
        '''
        Return the result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        activity_arr = np.tanh(x)
        self.__activity_arr_list.append(activity_arr)
        if len(self.__activity_arr_list) > self.__memory_len:
            self.__activity_arr_list = self.__activity_arr_list[len(self.__activity_arr_list) - self.__memory_len:]
        return activity_arr

    def derivative(self, np.ndarray y):
        '''
        Return of derivative with respect to this activation function.

        Args:
            y   The result of activation.

        Returns:
            The result.
        '''
        activity_arr = self.__activity_arr_list.pop(-1)
        return y * (1 - activity_arr ** 2)
