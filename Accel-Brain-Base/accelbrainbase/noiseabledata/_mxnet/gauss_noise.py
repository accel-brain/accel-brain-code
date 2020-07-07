# -*- coding: utf-8 -*-
from accelbrainbase.noiseabledata.gauss_noise import GaussNoise as _GaussNoise
import mxnet.ndarray as nd


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

    def noise(self, arr, F=nd):
        '''
        Noise.

        Args:
            arr:    `mx.nd.array` or `mx.sym.array`.
            F:      `mx.ndarray` or `mx.symbol`.
        
        Returns:
            `mx.nd.array` or `mx.sym.array`.
        '''
        return arr + F.random.normal_like(data=arr, loc=self.__mu, scale=self.__sigma)
