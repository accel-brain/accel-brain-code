# -*- coding: utf-8 -*-
from accelbrainbase.noiseabledata.gauss_noise import GaussNoise as _GaussNoise
import torch


class GaussNoise(_GaussNoise):
    '''
    Gauss noise function.
    '''

    def __init__(self, mu=0.0, sigma=1.0):
        '''
        Init.

        Args:
            mu:     Mean of the Gauss distribution.
            sigma:  Standard deviation of the Gauss distribution.
        '''
        self.__mu = mu
        self.__sigma = sigma

    def noise(self, arr, F=torch):
        '''
        Noise.

        Args:
            arr:    `mx.nd.array` or `mx.sym.array`.
            F:      `mx.ndarray` or `mx.symbol`.
        
        Returns:
            `mx.nd.array` or `mx.sym.array`.
        '''
        noise_arr = torch.normal(
            mean=self.__mu, 
            std=self.__sigma,
            size=arr.shape
        )
        noise_arr = noise_arr.to(arr.device)
        return arr + noise_arr
