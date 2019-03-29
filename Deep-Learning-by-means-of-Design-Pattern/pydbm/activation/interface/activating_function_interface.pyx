# -*- coding: utf-8 -*-
from pydbm.optimization.batch_norm import BatchNorm
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod


class ActivatingFunctionInterface(metaclass=ABCMeta):
    '''
    Abstract class for building activation functions.

    Two distinctions are introduced in this class design.

    What was first introduced is the distinction between 
    an activate in forward propagation and a derivative in back propagation.
    This two kind of methods enable implementation of learning algorithm 
    based on probabilistic gradient descent method etc, in relation to 
    the neural networks theory.

    The second distinction corresponds to the difference based on 
    the presence or absence of memory retention.
    In `activate` and `derivative`, the memories of propagated data points
    will be stored for computing **delta**. On the other hand, in `forward` and `backword`,
    the memories will be not stored.

    The methods that can perform forward and back propagation independently of the recording 
    for delta calculations are particularly useful for models such as `ConvolutionalAutoEncoder` 
    that perform deconvolution as transposition.
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
        Activate and extract feature points in forward propagation.

        Args:
            x   `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of the activated feature points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, np.ndarray y):
        '''
        Derivative and extract delta in back propagation.

        Args:
            y:  `np.ndarray` of delta.

        Returns:
            `np.ndarray` of delta.
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
            y:  `np.ndarray` of delta.

        Returns:
            The result.
        '''
        raise NotImplementedError()
