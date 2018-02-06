# -*- coding: utf-8 -*-
import random
from pydbm_mxnet.neuron_object import Neuron
from pydbm_mxnet.neuron.interface.visible_layer_interface import VisibleLayerInterface
from pydbm_mxnet.neuron.interface.hidden_layer_interface import HiddenLayerInterface
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


class FeaturePointNeuron(Neuron, VisibleLayerInterface, HiddenLayerInterface):
    '''
    For considering the feature points as `virtual` observed data points,
    instantiate neurons in hidden layer as the neurons in visible layer.

    This object is functionally equivalent to neurons in hidden layer and visible layer.
    '''

    # The object of neurons in visible layer.
    __visible_layer_interface = None

    def get_activating_function(self):
        ''' getter of activating_function '''
        if isinstance(self.__visible_layer_interface.activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__visible_layer_interface.activating_function

    def set_activating_function(self, value):
        ''' setter of activating_function '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__visible_layer_interface.activating_function = value

    activating_function = property(get_activating_function, set_activating_function)

    def __init__(self, visible_layer_interface):
        '''
        Initialize `self` as not only the instance of `visible_layer_interface` 
        but also inheritance of `visible_layer_interface`
        so that `self` can activate as neurons in visible layer.

        Args:
            visible_layer_interface:    The object of neurons.
        '''
        self.__visible_layer_interface = visible_layer_interface

    def observe_data_point(self, x):
        '''
        Input observed data points.
        
        Args:
            x:  observed data points.
        '''
        self.__visible_layer_interface.observe_data_point(x)
        self.activity = x

    def visible_update_state(self, link_value):
        '''
        Update activity as neurons in visible layer.

        Args:
            link_value:      Input value.

        '''
        self.__visible_layer_interface.visible_update_state(link_value)
        self.activity = self.__visible_layer_interface.activity

    def hidden_update_state(self, link_value):
        '''
        Update activity as neurons in hidden layer.

        Args:
            link_value:      Input value.

        '''
        self.visible_update_state(link_value)

    def update_bias(self, learning_rate):
        '''
        Update biases.

        Args:
            learning_rate:  Learning rate.
        '''
        diff_bias = learning_rate * self.activity
        self.__visible_layer_interface.diff_bias += diff_bias
        self.diff_bias += diff_bias
