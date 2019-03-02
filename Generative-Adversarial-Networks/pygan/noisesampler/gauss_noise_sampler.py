# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler


class GaussNoiseSampler(NoiseSampler):
    '''
    Generate samples based on the noise prior by Gauss distribution.
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

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        generated_arr = np.random.normal(loc=self.__mu, scale=self.__sigma, size=self.__output_shape)
        if self.noise_sampler is not None:
            generated_arr += self.noise_sampler.generate()
        
        return generated_arr
