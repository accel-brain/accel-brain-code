# -*- coding: utf-8 -*-
from accelbrainbase.noiseabledata.cutout_noise import CutoutNoise as _CutoutNoise
import numpy as np
import mxnet.ndarray as nd


class CutoutNoise(_CutoutNoise):
    '''
    Gauss noise function.
    '''

    def __init__(self, height=10, width=10):
        '''
        Init.

        Args:

        '''
        self.__height = height
        self.__width = width

    def noise(self, arr, F=nd):
        '''
        Noise.

        Args:
            arr:    Tensor (4 or 5 rank).
        
        Returns:
            Tensor.
        '''
        section_arr = F.ones_like(arr)
        if section_arr.ndim == 4:
            start_hight = np.random.randint(low=0, high=section_arr.shape[2] - self.__height)
            start_width = np.random.randint(low=0, high=section_arr.shape[3] - self.__width)
            section_arr[:, :, start_hight:start_hight+self.__height, start_width:start_width+self.__width] = 0
        elif section_arr.ndim == 5:
            start_hight = np.random.randint(low=0, high=section_arr.shape[3] - self.__height)
            start_width = np.random.randint(low=0, high=section_arr.shape[4] - self.__width)
            section_arr[:, :, :, start_hight:start_hight+self.__height, start_width:start_width+self.__width] = 0
        return arr * section_arr
