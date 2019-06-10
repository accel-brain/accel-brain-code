# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float64_t DOUBLE_t


class ParamsInitializer(object):
    '''
    Params Initializer.
    '''

    def __init__(self, sampler_f=np.random.normal, astype=np.float32):
        '''
        Init.

        Args:
            sampler:        A function of a Random sampling.
            astype:         Type of parameters.
        '''
        self.__sampler_f = sampler_f
        self.__astype = astype

    def sample(self, size, **kwargs):
        '''
        Random sampling.

        Args:
            size:           `int` or `tuple` of `int` of output shape.
            **kwargs:       Parameters other than `size` to be input to function `sample_f`.
        
        Returns:
            Returns by `sample_f`.
        '''
        cdef np.ndarray params_arr = self.__sampler_f(size=size, **kwargs)
        params_arr = params_arr.astype(self.__astype)
        return params_arr
