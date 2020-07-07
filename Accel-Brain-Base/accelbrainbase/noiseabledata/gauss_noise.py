# -*- coding: utf-8 -*-
from accelbrainbase.noiseable_data import NoiseableData
import numpy as np


class GaussNoise(NoiseableData):
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

    def noise(self, arr):
        '''
        Noise.

        Args:
            F:      `mx.ndarray` or `mx.symbol`.
            arr:    `mx.nd.array` or `mx.sym.array`.
        
        Returns:
            `mx.nd.array` or `mx.sym.array`.
        '''
        return arr + npz.random.normal(
            size=arr.shape, 
            loc=self.__mu, 
            scale=self.__sigma
        )
