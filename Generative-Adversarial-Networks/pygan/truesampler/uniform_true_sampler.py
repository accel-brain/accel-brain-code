# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler


class UniformSampler(TrueSampler):
    '''
    Generate samples based on the noise prior by Uniform distribution.
    '''

    def __init__(self, low, high, output_shape):
        '''
        Init.

        Args:
            low:            Lower boundary of the output interval.
                            All values generated will be greater than or equal to low. 

            high:           Upper boundary of the output interval.
                            All values generated will be less than high.

            output_shape:   Output shape.
                            the shape is `(batch size, d1, d2, d3, ...)`.
        '''
        self.__low = low
        self.__high = high
        self.__output_shape = output_shape

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        return np.random.uniform(loc=self.__low, scale=self.__high, size=self.__output_shape)
