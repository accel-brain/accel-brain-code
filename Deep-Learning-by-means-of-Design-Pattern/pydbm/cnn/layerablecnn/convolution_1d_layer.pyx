# -*- coding: utf-8 -*-
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class Convolution1DLayer(LayerableCNN):
    '''
    1-D Convolution Layer.

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
        Forward propagation in CNN layers.
        
        Override.

        Args:
            observed_arr:      2-rank array like or sparse matrix.
        
        Returns:
            2-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] result_arr = self.convolve(observed_arr)
        return self.graph.activation_function.activate(result_arr)

    def convolve(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr, no_bias_flag=False):
        '''
        Convolution.
        
        Args:
            observed_arr:   2-rank array like or sparse matrix.
            no_bias_flag:   Use bias or not.
        
        Returns:
            2-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=2] result_arr = np.empty_like(observed_arr)
        cdef int row = observed_arr.shape[0]
        for row in range(observed_arr.shape[0]):
            result_arr[row] = np.convolve(
                observed_arr[row], 
                v=self.graph.weight_arr, 
                mode="same"
            )
        if no_bias_flag is False:
            result_arr += self.graph.bias_arr

        return result_arr

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
        delta_arr = self.deconvolve(delta_arr)
        weight_arr = np.array([self.graph.weight_arr] * delta_arr.shape[0])

        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = delta_arr.mean(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_weight_arr = np.dot(weight_arr.T, delta_arr).mean(axis=1)
        if self.__delta_bias_arr is None:
            self.__delta_bias_arr = delta_bias_arr
        else:
            self.__delta_bias_arr += delta_bias_arr

        if self.__delta_weight_arr is None:
            self.__delta_weight_arr = delta_weight_arr
        else:
            self.__delta_weight_arr += delta_weight_arr

        return delta_arr

    def deconvolve(self, np.ndarray[DOUBLE_t, ndim=2] delta_arr):
        '''
        Deconvolution also called transposed convolutions
        "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

        Args:
            delta_arr:    2-rank array like or sparse matrix.

        Returns:
            2-rank array like or sparse matrix.,

        References:
            - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

        '''
        return self.convolve(delta_arr, no_bias_flag=True)

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
