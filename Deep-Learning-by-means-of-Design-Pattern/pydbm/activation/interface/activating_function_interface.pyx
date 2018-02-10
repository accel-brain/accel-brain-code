# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ActivatingFunctionInterface(metaclass=ABCMeta):
    '''
    Interface of activation functions.
    '''

    @abstractmethod
    def activate(self, x):
        '''
        Return of result from this activation function.

        Args:
            x   Parameter.

        Returns:
            The result.
        '''
        raise NotImplementedError()
