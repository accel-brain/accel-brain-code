# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler
from pydbm.cnn.featuregenerator.image_generator import ImageGenerator


class SineWaveSampler(TrueSampler):
    '''
    Generate samples based on the noise prior by sine wave distribution.
    '''

    def __init__(
        self,
        batch_size,
        seq_len,
        amp=0.5,
        sampling_freq=8000,
        freq=440,
        sec=5,
        norm_mode="z_score"
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size.
            seq_len:                        The length of sequences.
            amp:                            Amp.
            sample_freq:                    Sample frequency.
            freq:                           Frequency.
            sec:                            Second.
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.

        '''
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__amp = amp
        self.__sampling_freq = sampling_freq
        self.__freq = freq
        self.__sec = sec

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        for row in range(self.__batch_size):
            if observed_arr is None:
                observed_arr = self.__generate_sin(
                    amp=self.__amp,
                    sampling_freq=self.__sampling_freq,
                    freq=self.__freq,
                    sec=self.__sec,
                    seq_len=self.__seq_len
                )
            else:
                observed_arr = np.stack(
                    [
                        observed_arr,
                        self.__generate_sin(
                            amp=self.__amp,
                            sampling_freq=self.__sampling_freq,
                            freq=self.__freq,
                            sec=self.__sec,
                            seq_len=self.__seq_len
                        )
                    ],
                    axis=0
                )

        return observed_arr

    def __generate_sin(self, amp=0.5, sampling_freq=8000, freq=440, sec=5, seq_len=100):
        sin_list = []
        for n in np.arange(sampling_freq * sec):
            sin = amp * np.sin(2.0 * np.pi * freq * n / sampling_freq)
            sin_list.append(sin)
            if len(sin_list) > seq_len:
                break
        return np.array(sin_list[:seq_len])
