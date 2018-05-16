# -*- coding: utf-8 -*-
import numpy as np
cimport numpy as np
cimport cython
from pydbm.synapse.recurrent_temporal_graph import RecurrentTemporalGraph


class RNNGraph(RecurrentTemporalGraph):
    '''
    Recurrent Neural Network Restricted Boltzmann Machines (RNN-RBM)
    based on Complete Bipartite Graph.
    
    The shallower layer is to the deeper layer what the visible layer is to the hidden layer.
    '''

    __v_hat_weights_arr = np.array([])
    
    def get_v_hat_weights_arr(self):
        ''' getter '''
        if isinstance(self.__v_hat_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__v_hat_weights_arr

    def set_v_hat_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__v_hat_weights_arr = value
    
    v_hat_weights_arr = property(get_v_hat_weights_arr, set_v_hat_weights_arr)

    __hat_weights_arr = np.array([])
    
    def get_hat_weights_arr(self):
        ''' getter '''
        if isinstance(self.__hat_weights_arr, np.ndarray) is False:
            raise TypeError()
        return self.__hat_weights_arr

    def set_hat_weights_arr(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__hat_weights_arr = value

    hat_weights_arr = property(get_hat_weights_arr, set_hat_weights_arr)

    __rnn_hidden_bias = np.array([])
    
    def get_rnn_hidden_bias(self):
        ''' getter '''
        if isinstance(self.__rnn_hidden_bias, np.ndarray) is False:
            raise TypeError()
        return self.__rnn_hidden_bias

    def set_rnn_hidden_bias(self, value):
        ''' setter '''
        if isinstance(value, np.ndarray) is False:
            raise TypeError()
        self.__rnn_hidden_bias = value
    
    rnn_hidden_bias = property(get_rnn_hidden_bias, set_rnn_hidden_bias)
