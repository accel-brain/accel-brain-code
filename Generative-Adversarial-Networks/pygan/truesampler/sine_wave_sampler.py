# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler


class SineWaveSampler(TrueSampler):
    '''
    Sampler which draws samples from the `true` sine wave distribution.
    '''

    def __init__(
        self,
        batch_size,
        seq_len,
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
        self.__amp = amp
        self.__sampling_freq = sampling_freq
        self.__freq = freq
        self.__sec = sec
        self.__mu = mu
        self.__sigma = sigma

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = None
        for row in range(self.__batch_size):
            arr = self.__generate_sin(
                amp=self.__amp,
                sampling_freq=self.__sampling_freq,
                freq=self.__freq,
                sec=self.__sec,
                seq_len=self.__seq_len
            )
            arr = np.expand_dims(arr, axis=0)

            if observed_arr is None:
                observed_arr = arr
            else:
                observed_arr = np.r_[observed_arr, arr]

        observed_arr = np.expand_dims(observed_arr, axis=-1)
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
