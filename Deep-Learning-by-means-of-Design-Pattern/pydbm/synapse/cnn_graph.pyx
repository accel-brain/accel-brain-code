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
