# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy
import random


class Synapse(object):
    '''
    Synapse.
    '''

    # The list of nuron's object in shallowr layer.
    __shallower_neuron_arr = np.array([])
    # The list of neuron's object in deeper layer.
    __deeper_neuron_arr = np.array([])
    # `nd.array` of the weights.
    __weights_arr = np.array([])
    # `nd.array` of the difference of weights.
    __diff_weights_arr = np.array([])

    def get_shallower_neuron_arr(self):
        ''' getter '''
        return self.__shallower_neuron_arr

    def set_shallower_neuron_arr(self, numpy.ndarray value):
        ''' setter '''
        self.__shallower_neuron_arr = value

    def get_deeper_neuron_arr(self):
        ''' getter '''
        return self.__deeper_neuron_arr

    def set_deeper_neuron_arr(self, numpy.ndarray value):
        ''' setter '''
        self.__deeper_neuron_arr = value

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

    shallower_neuron_arr = property(get_shallower_neuron_arr, set_shallower_neuron_arr)
    deeper_neuron_arr = property(get_deeper_neuron_arr, set_deeper_neuron_arr)
    weights_arr = property(get_weights_arr, set_weights_arr)
    diff_weights_arr = property(get_diff_weights_arr, set_diff_weights_arr)

    def create_node(
        self,
        numpy.ndarray shallower_neuron_arr,
        numpy.ndarray deeper_neuron_arr,
        numpy.ndarray weights_arr=np.array([])
    ):
        '''
        Set links of nodes to the graphs.

        Args:
            shallower_neuron_arr:      The list of neuron's object in shallowr layer.
            deeper_neuron_arr:         The list of neuron's object in deeper layer.
            weights_arr:                `nd.array` of the weights.
        '''
        self.__shallower_neuron_arr = shallower_neuron_arr
        self.__deeper_neuron_arr = deeper_neuron_arr

        cdef numpy.ndarray init_weights_arr = np.random.rand(len(shallower_neuron_arr), len(deeper_neuron_arr))
        if weights_arr.shape[0]:
            self.__weights_arr = weights_arr
        else:
            self.__weights_arr = init_weights_arr

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
        cdef numpy.ndarray visible_activity_arr
        visible_activity_list = [self.shallower_neuron_arr[i].activity for i in range(len(self.shallower_neuron_arr))]
        if len(visible_activity_list) > 1 and sum(visible_activity_list) != 0:
            visible_activity_arr = np.array(visible_activity_list)
            visible_activity_arr = visible_activity_arr / visible_activity_arr.sum()

        cdef int k
        for k in range(visible_activity_arr.shape[0]):
            self.shallower_neuron_arr[k].activity = visible_activity_arr[k]

    def normalize_hidden_bias(self):
        '''
        normalize the neuron's activity in hidden layers.
        '''
        cdef int i
        cdef numpy.ndarray hidden_activity_arr
        hidden_activity_list = [self.deeper_neuron_arr[i].activity for i in range(self.deeper_neuron_arr.shape[0])]
        if len(hidden_activity_list) > 1 and sum(hidden_activity_list) != 0:
            hidden_activity_arr = np.array(hidden_activity_list)
            hidden_activity_arr = hidden_activity_arr / hidden_activity_arr.sum()

        cdef int k
        for k in range(hidden_activity_arr.shape[0]):
            self.deeper_neuron_arr[k].activity = hidden_activity_arr[k]
