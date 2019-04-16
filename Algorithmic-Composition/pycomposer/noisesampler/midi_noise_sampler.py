# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler


class MidiNoiseSampler(NoiseSampler):
    '''
    Generate samples based on the noise prior by Gauss distribution.
    '''

    def __init__(
        self, 
        batch_size=20,
        seq_len=10, 
        min_pitch=24,
        max_pitch=108

    ):
        '''
        Init.

        Args:
            batch_size:         Batch size.
            seq_len:            The length of sequneces.
                                The length corresponds to the number of `time` splited by `time_fraction`.

            min_pitch:          The minimum of note number.
            max_pitch:          The maximum of note number.

        '''
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__dim = max_pitch - min_pitch

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        generated_arr = np.random.uniform(
            low=0.1,
            high=0.9,
            size=((self.__batch_size, self.__seq_len, self.__dim))
        )

        if self.noise_sampler is not None:
            self.noise_sampler.output_shape = generated_arr.shape
            generated_arr += self.noise_sampler.generate()

        return generated_arr
