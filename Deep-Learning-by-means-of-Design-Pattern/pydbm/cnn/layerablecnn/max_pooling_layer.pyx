# -*- coding: utf-8 -*-
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class MaxPoolingLayer(LayerableCNN):
    '''
    Max Pooling Layer.
    '''
    
    __pool_height = 3
    __pool_width = 3
    __stride = 1
    __pad = 1
    
    __delta_weight_arr = np.array([])
    __delta_bias_arr = np.array([])

    def __init__(
        self,
        graph,
        int pool_height=3,
        int pool_width=3
    ):
        '''
        Init.
        
        Args:
            graph:          is-a `Synapse`.
            pool_heigh:     Height of pool.
            pool_width:     Width of pool.

        '''
        if isinstance(graph, Synapse):
            self.__graph = graph
        else:
            raise TypeError()

        self.__pool_height = pool_height
        self.__pool_width = pool_width
        self.__stride = graph.stride
        self.__pad = graph.pad

        self.__graph.constant_flag = True

    def convolve(self, np.ndarray[DOUBLE_t, ndim=4] img_arr, no_bias_flag=False):
        '''
        Convolution.
        
        Args:
            img_arr:        4-rank array like or sparse matrix.
            no_bias_flag:   Use bias or not.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        return self.forward_propagate(img_arr)

    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            matriimg_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]

        cdef int result_height = int(1 + (img_height - self.__pool_height) / self.__stride)
        cdef int result_width = int(1 + (img_width - self.__pool_width) / self.__stride)
        
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr = self.affine_to_matrix(
            img_arr,
            self.__pool_height, 
            self.__pool_width, 
            self.__stride, 
            self.__pad
        )
        reshaped_img_arr = reshaped_img_arr.reshape(-1, self.__pool_height * self.__pool_width)
        cdef np.ndarray max_index_arr = reshaped_img_arr.argmax(axis=1)
        cdef np.ndarray result_arr = np.max(reshaped_img_arr, axis=1)
        cdef np.ndarray[DOUBLE_t, ndim=4] _result_arr = result_arr.reshape(
            img_sample_n,
            result_height,
            result_width,
            -1
        )
        _result_arr = _result_arr.transpose(0, 3, 1, 2)

        self.__img_arr = img_arr
        self.__max_index_arr = max_index_arr
        self.__channel = img_channel

        return _result_arr

    def deconvolve(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Deconvolution also called transposed convolutions
        "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

        Args:
            delta_arr:    4-rank array like or sparse matrix.

        Returns:
            Tuple data.
            - 4-rank array like or sparse matrix.,
            - 2-rank array like or sparse matrix.

        References:
            - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

        '''
        return self.back_propagate(delta_arr)

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.

        Args:
            delta_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_arr = delta_arr.transpose(0, 2, 3, 1)
        
        cdef int pool_shape = self.__pool_height * self.__pool_width
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_pool_arr = np.zeros((_delta_arr.size, pool_shape))
        cdef np.ndarray flatten_arr = self.__max_index_arr.flatten()
        delta_pool_arr[np.arange(self.__max_index_arr.size), flatten_arr] = _delta_arr.flatten()
        cdef int delta_row = _delta_arr.shape[0]
        cdef int delta_col = _delta_arr.shape[1]
        _shape = (delta_row, delta_col) + (pool_shape, )
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_pool_arr = delta_pool_arr.reshape(
            _shape[0],
            _shape[1],
            _shape[2],
            -1
        )

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_reshaped_pool_arr = _delta_pool_arr.reshape(
            _delta_pool_arr.shape[0] * _delta_pool_arr.shape[1] * _delta_pool_arr.shape[2], 
            -1
        )

        cdef int img_sample_n = delta_arr.shape[0]
        cdef int channel = self.__channel
        cdef int img_height = delta_arr.shape[2]
        cdef int img_width = delta_arr.shape[3]

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.affine_to_img(
            delta_reshaped_pool_arr,
            img_sample_n,
            channel,
            img_height,
            img_width, 
            self.__pool_height,
            self.__pool_width,
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


    __delta_weight_arr = None
    __delta_bias_arr = None

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
