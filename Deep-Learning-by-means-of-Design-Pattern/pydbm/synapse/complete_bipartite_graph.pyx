# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
cimport numpy
from pydbm.synapse_list import Synapse


class CompleteBipartiteGraph(Synapse):
    '''
    Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''

    def get_visible_neuron_arr(self):
        ''' getter '''
        return self.shallower_neuron_arr

    def set_visible_neuron_arr(self, value):
        ''' setter '''
        self.shallower_neuron_arr = value

    visible_neuron_arr = property(get_visible_neuron_arr, set_visible_neuron_arr)

    def get_hidden_neuron_arr(self):
        ''' getter '''
        return self.deeper_neuron_arr

    def set_hidden_neuron_arr(self, value):
        ''' setter '''
        self.deeper_neuron_arr = value

    hidden_neuron_arr = property(get_hidden_neuron_arr, set_hidden_neuron_arr)
    
    def update(self, double learning_rate):
        '''
        Update weights.

        Args:
            learning_rate:  Learning rate.
        '''
        cdef int i
        cdef numpy.ndarray visible_activity_arr = np.array([[self.visible_neuron_arr[i].activity] * len(self.hidden_neuron_arr) for i in range(len(self.visible_neuron_arr))])
        cdef int j
        cdef numpy.ndarray hidden_activity_arr = np.array([[self.hidden_neuron_arr[j].activity] * len(self.visible_neuron_arr) for j in range(len(self.hidden_neuron_arr))])
        self.diff_weights_arr = visible_activity_arr * hidden_activity_arr.T * learning_rate
