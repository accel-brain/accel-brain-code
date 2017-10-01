# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from pydbm.synapse_list import Synapse


class CompleteBipartiteGraph(Synapse):
    '''
    Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''

    def get_visible_neuron_list(self):
        ''' getter '''
        return self.shallower_neuron_list

    def set_visible_neuron_list(self, value):
        ''' setter '''
        self.shallower_neuron_list = value

    visible_neuron_list = property(get_visible_neuron_list, set_visible_neuron_list)

    def get_hidden_neuron_list(self):
        ''' getter '''
        return self.deeper_neuron_list

    def set_hidden_neuron_list(self, value):
        ''' setter '''
        self.deeper_neuron_list = value

    hidden_neuron_list = property(get_hidden_neuron_list, set_hidden_neuron_list)
    
    def update(self, double learning_rate):
        '''
        Update weights.

        Args:
            learning_rate:  Learning rate.
        '''
        cdef int i
        visible_activity_arr = np.array([[self.visible_neuron_list[i].activity] * len(self.hidden_neuron_list) for i in range(len(self.visible_neuron_list))])
        cdef int j
        hidden_activity_arr = np.array([[self.hidden_neuron_list[j].activity] * len(self.visible_neuron_list) for j in range(len(self.hidden_neuron_list))])
        self.diff_weights_arr = visible_activity_arr * hidden_activity_arr.T * learning_rate
