# -*- coding: utf-8 -*-
import pyximport; pyximport.install()
from abc import ABCMeta, abstractmethod


class ActivatingFunctionInterface(metaclass=ABCMeta):
    '''
    Interface of activation functions.
    '''

    @abstractmethod
    def activate(self, double x):
        '''
        Return of result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, double y):
        '''
        Derivative.

        Args:
            y:  Parameter.

        Returns:
            The result.
        '''
        raise NotImplementedError()
