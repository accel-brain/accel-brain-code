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

    # Activation function.
    __activation_function = None
    
    def get_activation_function(self):
        ''' getter '''
        return self.__activation_function
    
    def set_activation_function(self, value):
        ''' setter '''
        self.__activation_function = value
    
    activation_function = property(get_activation_function, set_activation_function)

    # Activation function for deconvolution.
    __deactivation_function = None
    
    def get_deactivation_function(self):
        ''' getter '''
        return self.__deactivation_function
    
    def set_deactivation_function(self, value):
        ''' setter '''
        self.__deactivation_function = value
    
    deactivation_function = property(get_deactivation_function, set_deactivation_function)

    def __init__(
        self,
        activation_function,
        deactivation_function=None,
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
            activation_function:    Activation function.
            deactivation_function:  Activation function for deconvolution.
            filter_num:             The number of kernels(filters).
            channel:                Channel of image files.
            kernel_size:            Size of the kernels.
            stride:                 Stride.
            pad:                    Padding.
            scale:                  Scale of filters.
        '''
        self.__activation_function = activation_function
        if deactivation_function is not None:
            self.__deactivation_function = deactivation_function
        else:
            from copy import deepcopy
            self.__deactivation_function = deepcopy(activation_function)

        self.__weight_arr = np.random.normal(size=(filter_num, channel, kernel_size, kernel_size)) * scale
        self.__bias_arr = np.zeros((filter_num, ))
        
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
