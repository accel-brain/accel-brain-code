# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
import random


class Synapse(object):
    '''
    Synapse.
    '''

    # The list of nuron's object in shallowr layer.
    __shallower_neuron_list = []
    # The list of neuron's object in deeper layer.
    __deeper_neuron_list = []
    # `nd.array` of the weights.
    __weights_arr = None
    # `nd.array` of the difference of weights.
    __diff_weights_arr = None

    def get_shallower_neuron_list(self):
        ''' getter '''
        return self.__shallower_neuron_list

    def set_shallower_neuron_list(self, value):
        ''' setter '''
        self.__shallower_neuron_list = value

    def get_deeper_neuron_list(self):
        ''' getter '''
        return self.__deeper_neuron_list

    def set_deeper_neuron_list(self, value):
        ''' setter '''
        self.__deeper_neuron_list = value

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

    shallower_neuron_list = property(get_shallower_neuron_list, set_shallower_neuron_list)
    deeper_neuron_list = property(get_deeper_neuron_list, set_deeper_neuron_list)
    weights_arr = property(get_weights_arr, set_weights_arr)
    diff_weights_arr = property(get_diff_weights_arr, set_diff_weights_arr)

    def create_node(self, shallower_neuron_list, deeper_neuron_list, weights_arr=None):
        '''
        Set links of nodes to the graphs.

        Args:
            shallower_neuron_list:      The list of neuron's object in shallowr layer.
            deeper_neuron_list:         The list of neuron's object in deeper layer.
            weights_arr:                `nd.array` of the weights.
        '''
        self.__shallower_neuron_list = shallower_neuron_list
        self.__deeper_neuron_list = deeper_neuron_list
        if weights_arr is not None:
            self.__weights_arr = weights_arr
        else:
            self.__weights_arr = np.random.rand(len(shallower_neuron_list), len(deeper_neuron_list))

    def learn_weights(self):
        '''
        Update the weights of links.
        '''
        self.weights_arr = self.weights_arr + self.diff_weights_arr
        self.diff_weights_arr = np.zeros(self.weights_arr.shape, dtype=float)

    def normalize_visible_bias(self):
        '''
        Normalize the neuron's activity in visible layers.
        '''
        cdef int i
        visible_activity_list = [self.shallower_neuron_list[i].activity for i in range(len(self.shallower_neuron_list))]
        if len(visible_activity_list) > 1 and sum(visible_activity_list) != 0:
            visible_activity_arr = np.array(visible_activity_list)
            visible_activity_arr = visible_activity_arr / visible_activity_arr.sum()
            visible_activity_list = list(visible_activity_arr)

        cdef int k
        for k in range(len(visible_activity_list)):
            self.shallower_neuron_list[k].activity = visible_activity_list[k]

    def normalize_hidden_bias(self):
        '''
        normalize the neuron's activity in hidden layers.
        '''
        cdef int i
        hidden_activity_list = [self.deeper_neuron_list[i].activity for i in range(len(self.deeper_neuron_list))]
        if len(hidden_activity_list) > 1 and sum(hidden_activity_list) != 0:
            hidden_activity_arr = np.array(hidden_activity_list)
            hidden_activity_arr = hidden_activity_arr / hidden_activity_arr.sum()
            hidden_activity_list = list(hidden_activity_arr)

        cdef int k
        for k in range(len(hidden_activity_list)):
            self.deeper_neuron_list[k].activity = hidden_activity_list[k]
