# -*- coding: utf-8 -*-
from abc import abstractmethod
from mxnet.initializer import Initializer


class InitializableParams(Initializer):
    '''
    The interface to Initializes weights.
    '''

    @abstractmethod
    def _init_weight(self, name, arr):
        '''
        Initialize parameters.

        Args:
            name:   `str` of parameter name.
            arr:    `mx.nd.array` or `mx.sym.array`.
        
        Returns:
            `mx.nd.array` or `mx.sym.array`.
        '''
        raise NotImplementedError()
