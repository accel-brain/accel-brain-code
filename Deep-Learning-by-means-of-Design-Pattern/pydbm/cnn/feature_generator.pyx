# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class FeatureGenerator(object):
    '''
    Feature generator.
    '''

    @abstractproperty
    def epochs(self):
        ''' Epochs of Mini-batch. '''
        raise NotImplementedError()

    @abstractproperty
    def batch_size(self):
        ''' Batch size of Mini-batch. '''
        raise NotImplementedError()
    
    @abstractmethod
    def generate(self):
        '''
        Generate feature points.
        
        Returns:
            The tuple of feature points.
            The shape is: (`Training data`, `Training label`, `Test data`, `Test label`).
        '''
        raise NotImplementedError()
