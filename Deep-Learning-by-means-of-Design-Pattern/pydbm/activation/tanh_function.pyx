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
        Activate and extract feature points in forward propagation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of the activated feature points.
        '''
        activity_arr = np.tanh(x)
        self.__activity_arr_list.append(activity_arr)
        if len(self.__activity_arr_list) > self.__memory_len:
            self.__activity_arr_list = self.__activity_arr_list[len(self.__activity_arr_list) - self.__memory_len:]

        if self.batch_norm is not None:
            activity_arr = self.batch_norm.forward_propagation(activity_arr)

        return activity_arr

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

        activity_arr = self.__activity_arr_list.pop(-1)
        return y * (1 - activity_arr ** 2)

    def forward(self, np.ndarray x):
        '''
        Forward propagation but not retain the activation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            The result.
        '''
        return np.tanh(x)

    def backward(self, np.ndarray y):
        '''
        Back propagation but not operate the activation.

        Args:
            y:  `np.ndarray` of delta.

        Returns:
            The result.
        '''
        activity_arr = self.__activity_arr_list[-1]
        return y * (1 - activity_arr ** 2)
