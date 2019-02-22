# -*- coding: utf-8 -*-
import numpy as np
from pygan.noise_sampler import NoiseSampler


class SineWaveSampler(NoiseSampler):
    '''
    Generate samples based on the noise prior by sine wave distribution.
    '''

    def __init__(
        self,
        batch_size,
        seq_len,
        dim=1,
        amp=0.5,
        sampling_freq=8000,
        freq=440,
        sec=5,
        mu=0.0,
        sigma=1.0,
        norm_mode="z_score"
    ):
        '''
        Init.

        Args:
            batch_size:                     Batch size.
            seq_len:                        The length of sequences.
            dim:                            The number of dimension of observed data points.
            amp:                            Amp.
            sample_freq:                    Sample frequency.
            freq:                           Frequency.
            sec:                            Second.
            mu:                             Mean of Gauss noise.
            sigma:                          STD of Gauss noise.
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.

        '''
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__dim = dim
        self.__amp = amp
        self.__sampling_freq = sampling_freq
        self.__freq = freq
        self.__sec = sec
        self.__mu = mu
        self.__sigma = sigma

    def generate(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        for row in range(self.__batch_size):
            arr = None
            for d in range(self.__dim):
                _arr = self.__generate_sin(
                    amp=self.__amp,
                    sampling_freq=self.__sampling_freq,
                    freq=self.__freq,
                    sec=self.__sec,
                    seq_len=self.__seq_len
                )
                _arr = np.expand_dims(_arr, axis=0)
                if arr is None:
                    arr = _arr
                else:
                    arr = np.r_[arr, _arr]

            arr = np.expand_dims(arr, axis=0)

            if observed_arr is None:
                observed_arr = arr
            else:
                observed_arr = np.r_[observed_arr, arr]

        if self.__dim == 1:
            observed_arr = np.expand_dims(observed_arr, axis=-1)
        
        observed_arr = observed_arr.transpose((0, 2, 1))
        gauss_noise = np.random.normal(loc=self.__mu, scale=self.__sigma, size=observed_arr.shape)
        return observed_arr + gauss_noise

    def __generate_sin(self, amp=0.5, sampling_freq=8000, freq=440, sec=5, seq_len=100):
        sin_list = []
        for n in np.arange(sampling_freq * sec):
            sin = amp * np.sin(2.0 * np.pi * freq * n / sampling_freq)
            sin_list.append(sin)
            if len(sin_list) > seq_len:
                break
        return np.array(sin_list[:seq_len])
