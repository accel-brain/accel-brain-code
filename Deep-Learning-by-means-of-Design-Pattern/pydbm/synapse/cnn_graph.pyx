# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse_list import Synapse


class CNNGraph(Synapse):
    '''
    Computation graph in CNN.
    '''
    
    # Weight matrix (kernel)
    __weight_arr = None
    
    def get_weight_arr(self):
        ''' getter '''
        return self.__weight_arr

    def set_weight_arr(self, value):
        ''' setter '''
        self.__weight_arr = value
    
    weight_arr = property(get_weight_arr, set_weight_arr)

    # Bias vector.
    __bias_arr = None
    
    def get_bias_arr(self):
        ''' getter '''
        return self.__bias_arr

    def set_bias_arr(self, value):
        ''' setter '''
        self.__bias_arr = value
    
    bias_arr = property(get_bias_arr, set_bias_arr)

    def __init__(
        self,
        int filter_num=30,
        int channel=3,
        int kernel_size=3,
        int stride=1,
        int pad=1,
        double scale=0.01
    ):
        '''
        Init.
        
        Args:
            filter_num:     The number of kernels(filters).
            channel:        Channel of image files.
            kernel_size:    Size of the filters.
            stride:         Stride.
            pad:            Padding.
            scale:          Scale of filters.
        '''
        self.__weight_arr = np.random.normal(size=(filter_num, channel, kernel_size, kernel_size)) * scale
        self.__bias_arr = np.random.normal(size=(filter_num, )) * scale
        
        self.__stride = stride
        self.__pad = pad

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_stride(self):
        ''' getter '''
        return self.__stride

    stride = property(get_stride, set_readonly)

    def get_pad(self):
        ''' getter '''
        return self.__pad

    pad = property(get_pad, set_readonly)
