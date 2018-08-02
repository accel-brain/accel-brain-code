# -*- codiAdam utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.rnn.optimization.opt_params import OptParams


class Adam(OptParams):
    '''
    Adam.
    '''

    def __init__(self, double beta_1=0.9, double beta_2=0.99):
        '''
        Init.
        
        Args:
            momentum:    Momentum.
        '''
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__first_moment_list = []
        self.__second_moment_list = []
        self.__epoch = 0

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

        if len(self.__first_moment_list) == 0 or len(self.__first_moment_list) != len(params_list):
            self.__first_moment_list  = [None] * len(params_list)
        if len(self.__second_moment_list) == 0 or len(self.__second_moment_list) != len(params_list):
            self.__second_moment_list = [None] * len(params_list)
        
        if self.__first_moment_list[0] is None or self.__second_moment_list[0] is None:
            self.__epoch = 0

        self.__epoch += 1

        learning_rate = learning_rate * np.sqrt(1 - self.__beta_2 ** self.__epoch) / (1 - self.__beta_1 ** self.__epoch)

        for i in range(len(params_list)):
            if params_list[i] is None or grads_list[i] is None:
                continue

            if self.__first_moment_list[i] is not None or self.__second_moment_list[i] is not None:
                self.__second_moment_list[i] += (1 - self.__beta_1) * (grads_list[i] - self.__second_moment_list[i])
                self.__first_moment_list[i] += (1 - self.__beta_2) * (grads_list[i] ** 2 - self.__first_moment_list[i])
            else:
                self.__second_moment_list[i] = (1 - self.__beta_1) * (grads_list[i])
                self.__first_moment_list[i] = (1 - self.__beta_2) * (grads_list[i] ** 2)

            params_list[i] -= learning_rate * self.__second_moment_list[i] / (np.sqrt(self.__first_moment_list[i] + 1e-15))

        return params_list
