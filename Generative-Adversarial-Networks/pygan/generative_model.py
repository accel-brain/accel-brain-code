# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class GenerativeModel(metaclass=ABCMeta):
    '''
    Sampler which draws samples from the `fake` distribution.
    '''

    @abstractproperty
    def noise_sampler(self):
        '''
        is-a `NoiseSampler`.
        '''
        raise NotImplementedError()

    @abstractmethod
    def draw(self):
        '''
        Draws samples from the `fake` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()

    @abstractmethod
    def learn(self, grad_arr):
        '''
        Update this Generator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        raise NotImplementedError()
