# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.optimization.opt_params import OptParams


class SGD(OptParams):
    '''
    Stochastic Gradient Descent.

    References:
        - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

    '''

    def __init__(self, double momentum=0.9):
        '''
        Init.
        
        Args:
            momentum:    Momentum.
        '''
        self.__momentum = momentum
        self.__variation_list = []
        logger = getLogger("pydbm")
        self.__logger = logger

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

        if self.__momentum > 0:
            if len(self.__variation_list) == 0 or len(self.__variation_list) != len(params_list):
                self.__variation_list  = [None] * len(params_list)

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
                        self.__variation_list[i] = self.__momentum * self.__variation_list[i] - learning_rate * grads_list[i]
                        self.__variation_list[i] = np.nan_to_num(self.__variation_list[i])
                        params_list[i] += self.__variation_list[i]
                    else:
                        params_list[i] -= np.nan_to_num(learning_rate * grads_list[i])
                        self.__variation_list[i] = learning_rate * grads_list[i]

                    if params_ndim > 2:
                        params_list[i] = params_list[i].reshape(params_shape)
                    if grads_ndim > 2:
                        grads_list[i] = grads_list[i].reshape(grads_shape)

                except:
                    self.__logger.debug("Exception raised (key: " + str(i) + ")")
                    raise

        else:
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

                    params_list[i] = np.nansum(
                        np.array([
                            np.expand_dims(params_list[i], axis=0),
                            np.nanprod(
                                np.array([
                                    np.expand_dims(np.ones_like(params_list[i]) * learning_rate, axis=0),
                                    np.expand_dims(grads_list[i], axis=0)
                                ]),
                                axis=0
                            )
                        ]),
                        axis=0
                    )[0]

                    if params_ndim > 2:
                        params_list[i] = params_list[i].reshape(params_shape)
                    if grads_ndim > 2:
                        grads_list[i] = grads_list[i].reshape(grads_shape)

                except:
                    self.__logger.debug("Exception raised (key: " + str(i) + ")")
                    raise

        return params_list
