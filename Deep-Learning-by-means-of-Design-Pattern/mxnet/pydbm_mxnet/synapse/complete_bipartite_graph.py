# -*- coding: utf-8 -*-
import mxnet as mx
from pydbm_mxnet.synapse_list import Synapse
from pydbm_mxnet.activation.interface.activating_function_interface import ActivatingFunctionInterface


class CompleteBipartiteGraph(Synapse):
    '''
    Complete Bipartite Graph.
    
    The visible layer is to the hidden layer what the visible layer is to the hidden layer.
    '''

    __visible_activity_arr = None

    def get_visible_activity_arr(self):
        ''' getter '''
        return self.__visible_activity_arr

    def set_visible_activity_arr(self, value):
        ''' setter '''
        self.__visible_activity_arr = value

    visible_activity_arr = property(get_visible_activity_arr, set_visible_activity_arr)

    __hidden_activity_arr = None

    def get_hidden_activity_arr(self):
        ''' getter '''
        return self.__hidden_activity_arr

    def set_hidden_activity_arr(self, value):
        ''' setter '''
        self.__hidden_activity_arr = value

    hidden_activity_arr = property(get_hidden_activity_arr, set_hidden_activity_arr)

    __visible_bias_arr = None

    def get_visible_bias_arr(self):
        ''' getter '''
        return self.__visible_bias_arr

    def set_visible_bias_arr(self, value):
        ''' setter '''
        self.__visible_bias_arr = value

    visible_bias_arr = property(get_visible_bias_arr, set_visible_bias_arr)

    __hidden_bias_arr = None

    def get_hidden_bias_arr(self):
        ''' getter '''
        return self.__hidden_bias_arr

    def set_hidden_bias_arr(self, value):
        ''' setter '''
        self.__hidden_bias_arr = value

    hidden_bias_arr = property(get_hidden_bias_arr, set_hidden_bias_arr)
    
    def get_visible_activating_function(self):
        ''' getter '''
        if isinstance(self.deeper_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.deeper_activating_function

    def set_visible_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.deeper_activating_function = value

    visible_activating_function = property(get_visible_activating_function, set_visible_activating_function)

    def get_hidden_activating_function(self):
        ''' getter '''
        if isinstance(self.shallower_activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.shallower_activating_function

    def set_hidden_activating_function(self, value):
        ''' setter '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.shallower_activating_function = value

    hidden_activating_function = property(get_hidden_activating_function, set_hidden_activating_function)

    def create_node(
        self,
        visible_neuron_count,
        hidden_neuron_count,
        shallower_activating_function,
        deeper_activating_function,
        weights_arr=None
    ):
        '''
        Set links of nodes to the graphs.

        Override.

        Args:
            shallower_neuron_count:         The number of neuron in shallowr layer.
            deeper_neuron_count:            The number of neuron in deeper layer.
            shallower_activating_function:  The activation function in shallower layer.
            deeper_activating_function:     The activation function in deeper layer.
            weights_arr:                    `nd.array` of the weights.
        '''
        self.visible_bias_arr = mx.ndarray.random_uniform(low=0, high=1, shape=(visible_neuron_count, ))
        self.hidden_bias_arr = mx.ndarray.random_uniform(low=0, high=1, shape=(hidden_neuron_count, ))

        super().create_node(
            visible_neuron_count,
            hidden_neuron_count,
            shallower_activating_function,
            deeper_activating_function,
            weights_arr
        )

    def update(self, learning_rate):
        '''
        Update weights.

        Args:
            learning_rate:  Learning rate.
        '''
        if isinstance(learning_rate, float) is False and isinstance(learning_rate, int) is False:
            raise TypeError()

        if self.diff_weights_arr is None:
            self.diff_weights_arr = self.visible_activity_arr * self.hidden_activity_arr.T * learning_rate
        else:
            self.diff_weights_arr += self.visible_activity_arr * self.hidden_activity_arr.T * learning_rate
