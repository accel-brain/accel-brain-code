# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class RegularizatableData(metaclass=ABCMeta):
    '''
    The interface to customize Regularizations.
    '''

    @abstractmethod
    def regularize(self, params_dict):
        '''
        Regularize parameters.

        Args:
            params_dict:    is-a `mxnet.gluon.ParameterDict`.
        
        Returns:
            `mxnet.gluon.ParameterDict`
        '''
        raise NotImplementedError()
