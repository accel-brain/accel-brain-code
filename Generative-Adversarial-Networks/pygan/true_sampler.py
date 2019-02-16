# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class TrueSampler(metaclass=ABCMeta):
    '''
    Sampler which draws samples from the `true` distribution.
    '''

    @abstractmethod
    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
