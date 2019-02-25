# -*- coding: utf-8 -*-
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class NNLayer(object):
    '''
    NN Layer.
    '''
    # Computation graph which is-a `Synapse`.
    __graph = None
    # Delta of weight matrix.
    __delta_weight_arr = np.array([[]])
    # Delta of bias vector.
    __delta_bias_arr = np.array([[]])

    def __init__(self, graph):
        '''
        Init.
        
        Args:
            graph:      is-a `Synapse`.
        '''
        if isinstance(graph, Synapse):
            self.__graph = graph
        else:
            raise TypeError()
        
        self.__delta_weight_arr = None
        self.__delta_bias_arr = None

    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN layers.
        
        Override.

        Args:
            observed_arr:      2-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr = self.graph.activation_function.activate(
            np.dot(observed_arr, self.graph.weight_arr) + self.graph.bias_arr
        )
        self.__observed_arr = observed_arr
        return pred_arr

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=2] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.

        Args:
            delta_arr:      2-rank array like or sparse matrix.
        
        Returns:
            2-rank array like or sparse matrix.
        '''
        delta_arr = self.graph.activation_function.derivative(delta_arr)
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = delta_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr = np.dot(
            self.__observed_arr.T,
            delta_arr
        )

        if self.__delta_bias_arr is None:
            self.__delta_bias_arr = delta_bias_arr
        else:
            self.__delta_bias_arr += delta_bias_arr

        if self.__delta_weight_arr is None:
            self.__delta_weight_arr = delta_weight_arr
        else:
            self.__delta_weight_arr += delta_weight_arr

        delta_arr = np.dot(delta_arr, self.graph.weight_arr.T)
        return delta_arr

    def reset_delta(self):
        '''
        Reset delta.
        '''
        self.delta_weight_arr = None
        self.delta_bias_arr = None

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_graph(self):
        ''' getter '''
        return self.__graph

    graph = property(get_graph, set_readonly)

    def get_delta_weight_arr(self):
        ''' getter '''
        return self.__delta_weight_arr

    def set_delta_weight_arr(self, value):
        ''' setter '''
        self.__delta_weight_arr = value

    delta_weight_arr = property(get_delta_weight_arr, set_delta_weight_arr)

    def get_delta_bias_arr(self):
        ''' getter '''
        return self.__delta_bias_arr

    def set_delta_bias_arr(self, value):
        ''' setter '''
        self.__delta_bias_arr = value

    delta_bias_arr = property(get_delta_bias_arr, set_delta_bias_arr)
