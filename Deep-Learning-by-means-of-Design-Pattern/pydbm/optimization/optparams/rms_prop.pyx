# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.optimization.opt_params import OptParams


class RMSProp(OptParams):
    '''
    Adaptive RootMean-Square (RMSProp) gradient decent algorithm.

    References:
        - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
    '''

    def __init__(self, decay_rate=0.99):
        '''
        Init.

        Args:
            decay_rate:     Decay rate.
        '''
        if decay_rate <= 0:
            raise ValueError("The value of `decay_rate` must be more than `0`.")

        self.__variation_list = []
        logger = getLogger("pydbm")
        self.__logger = logger
        self.__decay_rate = decay_rate

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

        cdef np.ndarray square_variation_arr = None

        for i in range(len(params_list)):
            try:
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

                if self.__variation_list[i] is not None:
                    self.__variation_list[i] = self.__variation_list[i] * self.__decay_rate
                    self.__variation_list[i] = self.__variation_list[i] + ((1 - self.__decay_rate) * np.square(np.nan_to_num(grads_list[i])))
                    self.__variation_list[i] = np.nan_to_num(self.__variation_list[i])
                    try:
                        square_variation_arr = np.nan_to_num(np.sqrt(self.__variation_list[i]))
                        square_variation_arr[square_variation_arr == 0] += 1e-08
                        params_list[i] = np.nan_to_num(
                            params_list[i] - ((learning_rate * np.nan_to_num(grads_list[i])) / square_variation_arr)
                        )
                    except:
                        print("-" * 100)
                        print(learning_rate)
                        print("-" * 100)
                        print(params_list[i])
                        print("-" * 100)
                        print(self.__variation_list[i])
                        print("-" * 100)
                        raise
                else:
                    params_list[i] -= np.nan_to_num(learning_rate * grads_list[i])
                    self.__variation_list[i] = np.square(np.nan_to_num(grads_list[i]))

                if params_ndim > 2:
                    params_list[i] = params_list[i].reshape(params_shape)
                if grads_ndim > 2:
                    grads_list[i] = grads_list[i].reshape(grads_shape)

            except:
                self.__logger.debug("Exception raised (key: " + str(i) + ")")
                raise

        return params_list
