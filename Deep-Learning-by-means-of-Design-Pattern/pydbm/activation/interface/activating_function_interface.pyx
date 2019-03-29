# -*- coding: utf-8 -*-
from pydbm.optimization.batch_norm import BatchNorm
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class ActivatingFunctionInterface(metaclass=ABCMeta):
    '''
    Interface of activation functions.
    '''
    
    # is-a `BatchNorm`.
    __batch_norm = None
    
    def get_batch_norm(self):
        ''' getter '''
        return self.__batch_norm
    
    def set_batch_norm(self, value):
        ''' setter '''
        if isinstance(value, BatchNorm) is False:
            raise TypeError()
        self.__batch_norm = value
    
    batch_norm = property(get_batch_norm, set_batch_norm)

    @abstractmethod
    def activate(self, np.ndarray x):
        '''
        Return of result from this activation function.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            The result.
        '''
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, np.ndarray y):
        '''
        Return of derivative result from this activation function.

        Args:
            y:                  `np.ndarray` of delta.

        Returns:
            The result.
        '''
        raise NotImplementedError()

    @abstractmethod
    def forward(self, np.ndarray x):
        '''
        Forward propagation but not retain the activation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            The result.
        '''
        raise NotImplementedError()

    @abstractmethod
    def backward(self, np.ndarray y):
        '''
        Back propagation but not operate the activation.

        Args:
            y:                  `np.ndarray` of delta.

        Returns:
            The result.
        '''
        raise NotImplementedError()
