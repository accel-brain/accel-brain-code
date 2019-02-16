# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class NoiseSampler(metaclass=ABCMeta):
    '''
    Generate samples based on the noise prior.
    '''

    @abstractmethod
    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
