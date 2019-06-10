# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse_list import Synapse
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.params_initializer import ParamsInitializer


class CNNOutputGraph(Synapse):
    '''
    Computation graph in CNN's output layers.
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
    __activating_function = None
    
    def get_activating_function(self):
        ''' getter '''
        if isinstance(self.__activating_function, ActivatingFunctionInterface) is False:
            raise TypeError("The type of `__activating_function` must be `ActivatingFunctionInterface`.")
        return self.__activating_function
    
    def set_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError("The type of `__activating_function` must be `ActivatingFunctionInterface`.")
        self.__activating_function = value
    
    activating_function = property(get_activating_function, set_activating_function)

    def __init__(
        self, 
        activating_function, 
        int hidden_dim, 
        int output_dim, 
        scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.
        
        Args:
            activating_function:    Activation function.
            hidden_dim:             Dimension in deepest hidden layer.
            output_dim:             Dimension in output layer.
            scale:                  Scale of parameters which will be `ParamsInitializer`.
            params_initializer:     is-a `ParamsInitializer`.
            params_dict:            `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        self.activating_function = activating_function
        self.__weight_arr = params_initializer.sample(
            size=(hidden_dim, output_dim),
            **params_dict
        ) * scale
        self.__bias_arr = np.zeros((output_dim, ))
