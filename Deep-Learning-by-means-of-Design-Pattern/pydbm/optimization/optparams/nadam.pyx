# -*- codiAdam utf-8 -*-
import numpy as np
cimport numpy as np
from pydbm.optimization.opt_params import OptParams


class Nadam(OptParams):
    '''
    Nesterov-accelerated Adaptive Moment Estimation (Nadam).

    References:
        - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
        - Dozat, T. (2016). Incorporating nesterov momentum into adam., Workshop track - ICLR 2016.
    '''

    def __init__(self, double beta_1=0.9, double beta_2=0.99, bias_corrected_flag=True):
        '''
        Init.
        
        Args:
            beta_1:                 Weight for frist moment parameters.
            beta_2:                 Weight for second moment parameters.
            bias_corrected_flag:    Compute bias-corrected moments or not.
        '''
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__second_moment_list = []
        self.__variation_list = []
        self.__bias_corrected_flag = bias_corrected_flag
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

        if len(self.__variation_list) == 0 or len(self.__variation_list) != len(params_list):
            self.__variation_list  = [self.__beta_1] * len(params_list)

        if len(self.__second_moment_list) == 0 or len(self.__second_moment_list) != len(params_list):
            for i in range(len(params_list)):
                second_moment_arr = np.zeros_like(params_list[i])
                if second_moment_arr.ndim > 2:
                    second_moment_arr = second_moment_arr.reshape((
                        second_moment_arr.shape[0],
                        -1
                    ))
                self.__second_moment_list.append(second_moment_arr)
        
        if self.__second_moment_list[0] is None:
            self.__epoch = 0

        self.__epoch += 1

        cdef np.ndarray momentum_arr
        cdef np.ndarray pre_variation_arr

        learning_rate = learning_rate * np.sqrt(1 - np.power(self.__beta_2, self.__epoch)) / (1 - np.power(self.__beta_1, self.__epoch))

        for i in range(len(params_list)):
            if params_list[i] is None or grads_list[i] is None:
                continue

            params_shape = params_list[i].shape
            params_ndim = params_list[i].ndim
            if params_ndim > 2:
                params_list[i] = params_list[i].reshape((
                    params_shape[0],
                    -1
                ))
            grads_shape = grads_list[i].shape
            grads_ndim = grads_list[i].ndim
            if grads_ndim > 2:
                grads_list[i] = grads_list[i].reshape((
                    grads_shape[0],
                    -1
                ))

            pre_variation_arr = self.__variation_list[i]
            self.__variation_list[i] = self.__variation_list[i] * self.__momentum
            self.__variation_list[i] = self.__variation_list[i] - (learning_rate * grads_list[i])

            momentum_arr = self.__variation_list[i] * self.__momentum / (1 - np.nanprod(self.__variation_list[i])) + ((1 - pre_variation_arr) / (1 - np.nanprod(pre_variation_arr)))

            self.__second_moment_list[i] = np.nan_to_num(
                (self.__beta_2 * self.__second_moment_list[i]) + (1 - self.__beta_2) * np.square(grads_list[i])
            )
            self.__second_moment_list[i] = np.nan_to_num(self.__second_moment_list[i])

            if self.__bias_corrected_flag is True:
                self.__second_moment_list[i] = self.__second_moment_list[i] / (1 - np.power(self.__beta_2, self.__epoch))

            var_arr = learning_rate * momentum_arr / (np.sqrt(self.__second_moment_list[i]) + 1e-08)
            params_list[i] = params_list[i] - np.nan_to_num(var_arr)
            
            self.__second_moment_list[i] = second_moment_arr

            if params_ndim > 2:
                params_list[i] = params_list[i].reshape(params_shape)
            if grads_ndim > 2:
                grads_list[i] = grads_list[i].reshape(grads_shape)

        return params_list
