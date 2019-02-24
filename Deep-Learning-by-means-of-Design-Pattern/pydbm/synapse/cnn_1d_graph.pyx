# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse_list import Synapse


class CNN1DGraph(Synapse):
    '''
    Computation graph in 1-D CNN.
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

    def __init__(
        self,
        activation_function,
        int kernel_size,
        int dimension,
        double scale=0.01
    ):
        '''
        Init.
        
        Args:
            activation_function:    Activation function.
            kernel_size:            Size of the kernel.
            dimension:              Dimension of feature points.
            scale:                  Scale of filters.
        '''
        self.__activation_function = activation_function
        self.__weight_arr = np.random.normal(size=kernel_size) * scale
        self.__bias_arr = np.zeros((dimension, ))
