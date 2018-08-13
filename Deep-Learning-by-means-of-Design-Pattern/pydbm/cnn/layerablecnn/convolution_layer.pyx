# -*- coding: utf-8 -*-
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ConvolutionLayer(LayerableCNN):
    '''
    Convolution Layer.
    '''
    # Computation graph which is-a `Synapse`.
    __graph = None
    # Delta of weight matrix.
    __delta_weight_arr = np.array([[]])
    # Delta of bias vector.
    __delta_bias_arr = np.array([[]])

    def __init__(self, graph, int stride=1, int pad=0):
        '''
        Init.
        
        Args:
            graph:      is-a `Synapse`.
            stride:     Stride.
            pad:        Padding.
        '''
        if isinstance(graph, Synapse):
            self.__graph = graph
        else:
            raise TypeError()

        self.__stride = stride
        self.__pad = pad
        
        self.__delta_weight_arr = None
        self.__delta_bias_arr = None

    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            matriimg_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef int sample_n = self.graph.weight_arr.shape[0]
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]

        if sample_n != img_sample_n:
            raise ValueError()

        cdef int result_h = int((img_height + 2 * self.__pad - kernel_height) / self.__stride) + 1
        cdef int result_w = int((img_width + 2 * self.__pad - kernel_width) / self.__stride) + 1

        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr = self.affine_to_matrix(
            img_arr,
            kernel_height, 
            kernel_width, 
            self.__stride, 
            self.__pad
        )
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_weight_arr = self.graph.weight_arr.reshape(sample_n, -1).T
        cdef np.ndarray[DOUBLE_t, ndim=2] result_arr = np.dot(
            reshaped_img_arr,
            reshaped_weight_arr
        ) + self.graph.bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] _result_arr = result_arr.reshape(sample_n, result_h, result_w, -1)
        _result_arr = _result_arr.transpose(0, 3, 1, 2)

        self.__img_arr = img_arr
        self.__reshaped_img_arr = reshaped_img_arr
        self.__reshaped_weight_arr = reshaped_weight_arr

        return _result_arr

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.

        Args:
            delta_arr:      4-rank array like or sparse matrix.
        
        Returns:
            3-rank array like or sparse matrix.
        '''
        sample_n = self.graph.weight_arr.shape[0]
        channel = self.graph.weight_arr.shape[1]
        kernel_height = self.graph.weight_arr.shape[2]
        kernel_width = self.graph.weight_arr.shape[3]

        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_arr = delta_arr.transpose(0, 2, 3, 1)
        cdef np.ndarray[DOUBLE_t, ndim=2] __delta_arr = _delta_arr.reshape(-1, sample_n)
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = __delta_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr = np.dot(self.__reshaped_img_arr.T, __delta_arr)
        delta_weight_arr = delta_weight_arr.transpose(1, 0)
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_weight_arr = delta_weight_arr.reshape(
            sample_n,
            channel,
            kernel_height,
            kernel_width
        )
        if self.__delta_bias_arr is None:
            self.__delta_bias_arr = delta_bias_arr
        else:
            self.__delta_bias_arr += delta_bias_arr

        if self.__delta_weight_arr is None:
            self.__delta_weight_arr = delta_weight_arr
        else:
            self.__delta_weight_arr += delta_weight_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_reshaped_img_arr = np.dot(__delta_arr, self.__reshaped_weight_arr.T)
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.affine_to_img(
            delta_reshaped_img_arr,
            self.__img_arr, 
            kernel_height, 
            kernel_width, 
            self.__stride, 
            self.__pad
        )
        return delta_img_arr

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

    delta_weight_arr = property(get_delta_weight_arr, set_readonly)

    def get_delta_bias_arr(self):
        ''' getter '''
        return self.__delta_bias_arr
    
    delta_bias_arr = property(get_delta_bias_arr, set_readonly)
