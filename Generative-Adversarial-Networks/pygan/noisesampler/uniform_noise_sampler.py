# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler


class UniformNoiseSampler(NoiseSampler):
    '''
    Generate samples based on the noise prior by Uniform distribution.
    '''

    def __init__(self, low, high, output_shape):
        '''
        Init.

        Args:
            low:            Lower boundary of the output interval.
                            All values generated will be greater than or equal to low. 
                            The default value is `0.0`.

            high:           Upper boundary of the output interval.
                            All values generated will be less than high.
                            The default value is `1.0`.

            output_shape:   Output shape.
                            the shape is `(batch size, d1, d2, d3, ...)`.
        '''
        self.__low = low
        self.__high = high
        self.__output_shape = output_shape

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        generated_arr = np.random.uniform(low=self.__low, high=self.__high, size=self.__output_shape)
        if self.noise_sampler is not None:
            generated_arr += self.noise_sampler.generate()
        return generated_arr
