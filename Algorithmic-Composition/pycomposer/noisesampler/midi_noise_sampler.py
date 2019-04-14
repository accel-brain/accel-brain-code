# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler


class MidiNoiseSampler(NoiseSampler):
    '''
    Generate samples based on the noise prior by Gauss distribution.
    '''

    def __init__(
        self, 
        batch_size=20
    ):
        '''
        Init.

        Args:
            velocity_low:           Lower boundary of the output interval of Uniform noise for velocity.
            velocity_high:          Higher boundary of the output interval of Uniform noise for velocity.
        '''
        self.__batch_size = batch_size

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        generated_arr = np.random.uniform(
            low=0.1,
            high=0.9,
            size=((self.__batch_size, 1, 12))
        )

        if self.noise_sampler is not None:
            self.noise_sampler.output_shape = generated_arr.shape
            generated_arr += self.noise_sampler.generate()

        return generated_arr
