# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class CostFunctionable(metaclass=ABCMeta):
    '''
    The interface of cost function in annealing.
    '''

    @abstractmethod
    def compute(self, x):
        '''
        Compute.

        Args:
            x:    var.
        
        Returns:
            Cost.
        '''
        raise NotImplementedError()
