# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.optimization.opt_params import OptParams


class SGD(OptParams):
    '''
    Stochastic Gradient Descent.
    '''

    def __init__(self, double momentum=0.9):
        '''
        Init.
        
        Args:
            momentum:    Momentum.
        '''
        self.__momentum = momentum
        self.__variation_list = []

    def optimize(self, params_list, grads_list, double learning_rate):
        '''
        Return of result from this optimization function.
        
        Override.

        Args:
            params_dict:    `list` of parameters.
            grads_list:     `list` of gradation.
            learning_rate:  Learning rate.

        Returns:
            `list` of optimized parameters.
        '''
        if len(params_list) != len(grads_list):
            raise ValueError("The row of `params_list` and `grads_list` must be equivalent.")

        if len(self.__variation_list) == 0 or len(self.__variation_list) != len(params_list):
            self.__variation_list  = [None] * len(params_list)

        for i in range(len(params_list)):
            if params_list[i] is None or grads_list[i] is None:
                continue

            if self.__variation_list[i] is not None:
                self.__variation_list[i] = self.__momentum * self.__variation_list[i] - learning_rate * grads_list[i]
                self.__variation_list[i] = np.nan_to_num(self.__variation_list[i])
                params_list[i] += self.__variation_list[i]
            else:
                params_list[i] -= np.nan_to_num(learning_rate * grads_list[i])
                self.__variation_list[i] = learning_rate * grads_list[i]

        return params_list
