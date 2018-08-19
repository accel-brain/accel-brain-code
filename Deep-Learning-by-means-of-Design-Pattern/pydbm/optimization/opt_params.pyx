# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class OptParams(metaclass=ABCMeta):
    '''
    Abstract class of optimization functions.
    '''
    
    # Regularization for weights matrix
    # to repeat multiplying the weights matrix and `0.9`
    # until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
    __weight_limit = 0.9
    
    # Probability of dropout.
    __dropout_rate = 0.5

    @abstractmethod
    def optimize(self, params_list, np.ndarray grads_arr, double learning_rate):
        '''
        Return of result from this concrete optimization function.

        Args:
            params_dict:    `list` of parameters.
            grads_arr:      `np.ndarray` of gradation.
            learning_rate:  Learning rate.

        Returns:
            `list` of optimized parameters.
        '''
        raise NotImplementedError()

    def constrain_weight(self, np.ndarray weight_arr):
        '''
        Regularization for weights matrix
        to repeat multiplying the weights matrix and `0.9`
        until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
    
        Args:
            weight_arr:       wegiht matrix.
        
        Returns:
            weight matrix.
        '''
        while np.sum(np.square(weight_arr)) > self.weight_limit:
            weight_arr = weight_arr * 0.9

        return weight_arr

    def dropout(self, np.ndarray activity_arr):
        '''
        Dropout.
        
        Args:
            activity_arr:    The state of units.
        
        Returns:
            The state of units.
        '''
        if self.dropout_rate == 0.0:
            return activity_arr

        cdef int row = activity_arr.shape[0]
        cdef int col
        cdef np.ndarray dropout_rate_arr

        if activity_arr.ndim == 1:
            dropout_rate_arr = np.random.binomial(n=1, p=1-self.dropout_rate, size=(row, )).astype(int)
            
        if activity_arr.ndim > 1:
            col = activity_arr.shape[1]
            dropout_rate_arr = np.random.binomial(n=1, p=1-self.dropout_rate, size=(row, col)).astype(int)

        activity_arr = activity_arr * dropout_rate_arr
        return activity_arr

    def get_weight_limit(self):
        ''' getter '''
        return self.__weight_limit

    def set_weight_limit(self, value):
        ''' setter '''
        if isinstance(value, float):
            self.__weight_limit = value
        else:
            raise TypeError()

    weight_limit = property(get_weight_limit, set_weight_limit)

    def get_dropout_rate(self):
        ''' getter '''
        return self.__dropout_rate

    def set_dropout_rate(self, value):
        ''' setter '''
        if isinstance(value, float):
            self.__dropout_rate = value
        else:
            raise TypeError()

    dropout_rate = property(get_dropout_rate, set_dropout_rate)
