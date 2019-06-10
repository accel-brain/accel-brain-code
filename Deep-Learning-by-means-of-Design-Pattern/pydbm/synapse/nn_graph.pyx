# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse_list import Synapse
from pydbm.params_initializer import ParamsInitializer


class NNGraph(Synapse):
    '''
    Computation graph in the perceptron or Neural Network.
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
        int hidden_neuron_count,
        int output_neuron_count,
        double scale=1.0,
        params_initializer=ParamsInitializer(),
        params_dict={"loc": 0.0, "scale": 1.0}
    ):
        '''
        Init.
        
        Args:
            activation_function:    Activation function.
            hidden_neuron_count:    The number of hidden units.
            output_neuron_count:    The number of output units.
            scale:                  Scale of parameters which will be `ParamsInitializer`.
            params_initializer:     is-a `ParamsInitializer`.
            params_dict:            `dict` of parameters other than `size` to be input to function `ParamsInitializer.sample_f`.
        '''
        if isinstance(params_initializer, ParamsInitializer) is False:
            raise TypeError("The type of `params_initializer` must be `ParamsInitializer`.")

        self.__activation_function = activation_function
        self.__weight_arr = params_initializer.sample(
            size=(hidden_neuron_count, output_neuron_count),
            **params_dict
         ) * scale
        self.__bias_arr = np.zeros((output_neuron_count, ))
