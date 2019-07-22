# -*- coding utf-8 -*-
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

    def __init__(self, double beta_1=0.9, double beta_2=0.99, bias_corrected_flag=False):
        '''
        Init.
        
        Args:
            beta_1:                 Weight for frist moment parameters.
            beta_2:                 Weight for second moment parameters.
            bias_corrected_flag:    Compute bias-corrected first moment / second raw moment estimate or not.
        '''
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__beta_1_arr = np.array([])
        self.__first_moment_list = []
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

        if len(self.__first_moment_list) == 0 or len(self.__first_moment_list) != len(params_list):
            for i in range(len(params_list)):
                first_moment_arr = np.zeros_like(params_list[i])
                if first_moment_arr.ndim > 2:
                    first_moment_arr = first_moment_arr.reshape((
                        first_moment_arr.shape[0],
                        -1
                    ))
                self.__first_moment_list.append(first_moment_arr)

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
            self.__beta_1_arr = np.array([])

        self.__epoch += 1
        self.__beta_1_arr = np.insert(self.__beta_1_arr, 0, np.power(self.__beta_1, self.__epoch))

        learning_rate = learning_rate * np.sqrt(1 - np.power(self.__beta_2, self.__epoch)) / (1 - np.power(self.__beta_1, self.__epoch))

        cdef momentum_arr
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

            if self.__beta_1_arr.shape[0] > 1:
                grads_list[i] = self.__multiply_arr_scalar(
                    arr=grads_list[i],
                    scalar=1 / (1 - np.nanprod(self.__beta_1_arr[1:]))
                )
            
            self.__first_moment_list[i] = self.__multiply_arr_scalar(
                scalar=self.__beta_1,
                arr=self.__first_moment_list[i]
            ) + self.__multiply_arr_scalar(
                scalar=1 - self.__beta_1,
                arr=grads_list[i]
            )
            momentum_arr = self.__multiply_arr_scalar(
                arr=self.__first_moment_list[i],
                scalar=1 / (1 - np.nanprod(self.__beta_1_arr))
            )

            self.__second_moment_list[i] = self.__multiply_arr_scalar(
                scalar=self.__beta_2,
                arr=self.__second_moment_list[i]
            ) + self.__multiply_arr_scalar(
                scalar=(1 - self.__beta_2),
                arr=np.square(grads_list[i])
            )
            if self.__bias_corrected_flag is True:
                self.__second_moment_list[i] = self.__multiply_arr_scalar(
                    arr=self.__second_moment_list[i],
                    scalar=1 / (1 - np.power(self.__beta_2, self.__epoch))
                )

            momentum_arr = self.__multiply_arr_scalar(
                scalar=(1 - self.__beta_1),
                arr=grads_list[i]
            ) + self.__multiply_arr_scalar(
                scalar=self.__beta_1_arr[0],
                arr=momentum_arr
            )

            var_arr = self.__multiply_arr_scalar(
                scalar=learning_rate,
                arr=self.__multiply_arrs(
                    momentum_arr,
                    1 / (np.sqrt(self.__second_moment_list[i]) + 1e-08)
                )
            )

            params_list[i] = params_list[i] - var_arr

            if params_ndim > 2:
                params_list[i] = params_list[i].reshape(params_shape)
            if grads_ndim > 2:
                grads_list[i] = grads_list[i].reshape(grads_shape)

        return params_list

    def __multiply_arr_scalar(self, np.ndarray arr, scalar):
        return np.nanprod([np.ones_like(arr) * scalar, arr], axis=0)

    def __multiply_arrs(self, *args):
        return np.nanprod(args, axis=0)

    def __multiply_scalars(self, *args):
        return np.nanprod(args, axis=0)
