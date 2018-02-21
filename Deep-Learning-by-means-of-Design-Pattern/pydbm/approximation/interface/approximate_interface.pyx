# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
cimport cython
ctypedef np.float64_t DOUBLE_t


class ApproximateInterface(metaclass=ABCMeta):
    '''
    The interface for function approximations.
    '''

    @abstractproperty
    def reconstruct_error_list(self):
        ''' Reconstruction error. '''
        raise NotImplementedError()

    @abstractmethod
    def approximate_learn(
        self,
        graph,
        double learning_rate,
        double dropout_rate,
        np.ndarray observed_data_arr,
        int traning_count=1000
    ):
        '''
        learning with function approximation.

        Args:
            graph:                Graph of neurons.
            learning_rate:        Learning rate.
            dropout_rate:         Dropout rate.
            observed_data_arr:    observed data points.
            traning_count:        Training counts.

        Returns:
            Graph of neurons.
        '''
        raise NotImplementedError()

    @abstractmethod
    def compute_reconstruct_error(
        self,
        np.ndarray[DOUBLE_t, ndim=1] observed_data_arr, 
        np.ndarray[DOUBLE_t, ndim=1] reconstructed_arr
    ):
        '''
        Compute reconstruction error rate.
        '''
        raise NotImplementedError()
