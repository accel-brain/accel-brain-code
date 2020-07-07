# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class NoiseableData(metaclass=ABCMeta):
    '''
    The interface to customize noising function for building Denoising Auto-Encoders.
    '''

    @abstractmethod
    def noise(self, arr):
        '''
        Noise.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        raise NotImplementedError()
