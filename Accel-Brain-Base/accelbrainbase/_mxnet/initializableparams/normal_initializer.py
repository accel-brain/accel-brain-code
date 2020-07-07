# -*- coding: utf-8 -*-
from mxnet.initializer import Initializer
import mxnet.ndarray as nd
from mxnet import gluon
import numpy as np


class NormalInitializer(Initializer):
    '''
    Initializes weights with the probability density function of the normal distribution.
    '''
    
    def __init__(self, mu=0.0, sigma=1.0, scale=1.0, init_dtype=np.float16):
        '''
        Init.

        Args:
            mu:                 Mean (“centre”) of the distribution.
            sigma:              Standard deviation (spread or “width”) of the distribution.
            init_dtype:         `np.dtype`.
            scale:              `float` of scaling factor for initial parmaeters.
        '''

        super(NormalInitializer, self).__init__(sigma=sigma, mu=mu, init_dtype=init_dtype)
        self.__sigma = sigma
        self.__mu = mu
        self.__init_dtype = init_dtype
        self.__scale = scale

    def _init_weight(self, name, arr):
        nd.random.normal(self.__mu, self.__sigma, out=arr)
        arr = arr.astype(self.__init_dtype)
        arr = arr * self.__scale
