# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy as np
cimport cython
import random
ctypedef np.float64_t DOUBLE_t


class Synapse(object):
    '''
    The object of synapse.
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

    def create_node(
        self,
        shallower_neuron_list,
        deeper_neuron_list,
        np.ndarray weights_arr=np.array([])
    ):
        '''
        Set links of nodes to the graphs.

        Args:
            shallower_neuron_list:      The list of neuron's object in shallowr layer.
            deeper_neuron_list:         The list of neuron's object in deeper layer.
            weights_arr:                `nd.array` of the weights.
        '''
        self.shallower_neuron_list = shallower_neuron_list
        self.deeper_neuron_list = deeper_neuron_list

        cdef np.ndarray init_weights_arr = np.random.rand(len(shallower_neuron_list), len(deeper_neuron_list))
        if weights_arr.shape[0]:
            self.weights_arr = weights_arr
        else:
            self.weights_arr = init_weights_arr

    def learn_weights(self):
        '''
        Update the weights of links.
        '''
        self.weights_arr = self.weights_arr + self.diff_weights_arr
        cdef int row = self.weights_arr.shape[0]
        cdef int col = self.weights_arr.shape[1]
        self.diff_weights_arr = np.zeros((row, col), dtype=float)

    def normalize_visible_bias(self):
        '''
        Normalize the neuron's activity in visible layers.
        '''
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1] visible_activity_arr
        cdef int i_row = len(self.shallower_neuron_list)
        visible_activity_list = [self.shallower_neuron_list[i].activity for i in range(i_row)]
        if len(visible_activity_list) > 1 and sum(visible_activity_list) != 0:
            visible_activity_arr = np.array(visible_activity_list)
            visible_activity_arr = visible_activity_arr / visible_activity_arr.sum()

        cdef int k
        cdef int k_row = visible_activity_arr.shape[0]
        for k in range(k_row):
            self.shallower_neuron_list[k].activity = visible_activity_arr[k]

    def normalize_hidden_bias(self):
        '''
        normalize the neuron's activity in hidden layers.
        '''
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1] hidden_activity_arr
        cdef int i_row = len(self.deeper_neuron_list)
        hidden_activity_list = [self.deeper_neuron_list[i].activity for i in range(i_row)]
        if len(hidden_activity_list) > 1 and sum(hidden_activity_list) != 0:
            hidden_activity_arr = np.array(hidden_activity_list)
            hidden_activity_arr = hidden_activity_arr / hidden_activity_arr.sum()

        cdef int k
        for k in range(hidden_activity_arr.shape[0]):
            self.deeper_neuron_list[k].activity = hidden_activity_arr[k]
