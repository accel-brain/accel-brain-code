# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler


class GaussSampler(TrueSampler):
    '''
    Sampler which draws samples from the `true` Gauss distribution.
    '''

    def __init__(self, mu, sigma, output_shape):
        '''
        Init.

        Args:
            mu:             `float` or `array_like of floats`.
                            Mean (`centre`) of the distribution.

            sigma:          `float` or `array_like of floats`.
                            Standard deviation (spread or `width`) of the distribution.

            output_shape:   Output shape.
                            the shape is `(batch size, d1, d2, d3, ...)`.
        '''
        self.__mu = mu
        self.__sigma = sigma
        self.__output_shape = output_shape

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        return np.random.normal(loc=self.__mu, scale=self.__sigma, size=self.__output_shape)
