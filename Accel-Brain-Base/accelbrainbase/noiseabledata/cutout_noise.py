# -*- coding: utf-8 -*-
from accelbrainbase.noiseable_data import NoiseableData
import numpy as np


class CutoutNoise(NoiseableData):
    '''
    Gauss noise function.

    References:
        - DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552.
    '''

    def __init__(self, height=10, width=10):
        '''
        Init.

        Args:

        '''
        self.__height = height
        self.__width = width

    def noise(self, arr):
        '''
        Noise.

        Args:
            arr:    Tensor (4 or 5 rank).
        
        Returns:
            Tensor.
        '''
        section_arr = np.ones(arr.shape)
        if section_arr.ndim == 4:
            start_hight = np.random.randint(low=0, high=section_arr.shape[2] - self.__height)
            start_width = np.random.randint(low=0, high=section_arr.shape[3] - self.__width)
            section_arr[:, :, start_hight:start_hight+self.__height, start_width:start_width+self.__width] = 0
        elif section_arr.ndim == 5:
            start_hight = np.random.randint(low=0, high=section_arr.shape[3] - self.__height)
            start_width = np.random.randint(low=0, high=section_arr.shape[4] - self.__width)
            section_arr[:, :, :, start_hight:start_hight+self.__height, start_width:start_width+self.__width] = 0
        return arr * section_arr
