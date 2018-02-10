# -*- coding: utf-8 -*-
import random
import mxnet as mx
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


class Synapse(object):
    '''
    The object of synapse.
    '''
    # `mx.ndarray` of the weights.
    __weights_arr = None

    def get_weights_arr(self):
        ''' getter '''
        return self.__weights_arr

    def set_weights_arr(self, value):
        ''' setter '''
        self.__weights_arr = value

    def get_diff_weights_arr(self):
        ''' getter '''
        return self.__diff_weights_arr

    def set_diff_weights_arr(self, value):
        ''' setter '''
        self.__diff_weights_arr = value

    weights_arr = property(get_weights_arr, set_weights_arr)
    diff_weights_arr = property(get_diff_weights_arr, set_diff_weights_arr)

    __shallower_activating_function = None
    
    def get_shallower_activating_function(self):
        ''' getter '''
        if isinstance(self.__shallower_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__shallower_activating_function

    def set_shallower_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__shallower_activating_function = value

    shallower_activating_function = property(get_shallower_activating_function, set_shallower_activating_function)

    __deeper_activating_function = None

    def get_deeper_activating_function(self):
        ''' getter '''
        if isinstance(self.__deeper_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__deeper_activating_function

    def set_deeper_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__deeper_activating_function = value

    deeper_activating_function = property(get_deeper_activating_function, set_deeper_activating_function)

    def create_node(
        self,
        shallower_neuron_count,
        deeper_neuron_count,
        shallower_activating_function,
        deeper_activating_function,
        weights_arr=None,
    ):
        '''
        Set links of nodes to the graphs.

        Args:
            shallower_neuron_count:         The number of neuron in shallowr layer.
            deeper_neuron_count:            The number of neuron in deeper layer.
            shallower_activating_function:  The activation function in shallower layer.
            deeper_activating_function:     The activation function in deeper layer.
            weights_arr:                    `nd.array` of the weights.
        '''
        self.shallower_activating_function = shallower_activating_function
        self.deeper_activating_function = deeper_activating_function

        init_weights_arr = mx.ndarray.random.uniform(
            shape=(
                shallower_neuron_count,
                deeper_neuron_count
            )
        )
        if weights_arr is not None:
            self.weights_arr = weights_arr
        else:
            self.weights_arr = init_weights_arr

    def learn_weights(self):
        '''
        Update the weights of links.
        '''
        self.weights_arr = self.weights_arr + self.diff_weights_arr
        row = self.weights_arr.shape[0]
        col = self.weights_arr.shape[1]
        self.diff_weights_arr = mx.nd.zeros((row, col), dtype=float)
