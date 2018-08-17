# -*- coding: utf-8 -*-
from PIL import Image
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class DeconvolutionLayer(ConvolutionLayer):
    '''
    Deconvolution (transposed convolution) Layer.
    
    Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)
    
    So this class is sub class of `ConvolutionLayer`. The `DeconvolutionLayer` is-a `ConvolutionLayer`.
    
    Reference:
        Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

    '''
    # `Tuple` of target shape. The shape is (`height`, `width`).
    __target_shape = (10, 10)

    def __init__(self, graph):
        '''
        Init.
        
        Args:
            graph:          is-a `Synapse`.
        '''
        self.__stride = graph.stride
        self.__pad = graph.pad
        
        super().__init__(graph)
    
    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            img_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        if self.__target_shape is None:
            return super().forward_propagate(img_arr)

        cdef int sample_n = self.graph.weight_arr.shape[0]
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]

        cdef np.ndarray extend_weight_arr
        cdef np.ndarray circulant_arr
        cdef np.ndarray block_arr
        cdef np.ndarray reshaped_img_arr

        cdef int diff_height
        cdef int diff_width
        cdef np.ndarray resized_img_arr

        cdef np.ndarray[DOUBLE_t, ndim=4] result_arr = np.zeros((
            img_sample_n, 
            img_channel, 
            self.__target_shape[0], 
            self.__target_shape[1]
        ))
        for n in range(img_sample_n):
            for c in range(img_channel):
                img = Image.fromarray(img_arr[n, c])
                img = img.resize(self.__target_shape)
                resized_img_arr = np.asarray(img)
                result_arr[n, c] = resized_img_arr

        return super().forward_propagate(result_arr)

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.

        Args:
            delta_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        if self.__target_shape is None:
            return super().back_propagate(delta_arr)

        cdef int img_sample_n = delta_arr.shape[0]
        cdef int img_channel = delta_arr.shape[1]
        cdef int img_height = delta_arr.shape[2]
        cdef int img_width = delta_arr.shape[3]

        cdef np.ndarray resized_delta_arr

        cdef np.ndarray[DOUBLE_t, ndim=4] result_arr = np.zeros((
            img_sample_n, 
            img_channel, 
            self.__target_shape[0], 
            self.__target_shape[1]
        ))
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_arr = np.zeros((
            img_sample_n, 
            img_channel, 
            img_height, 
            img_width
        ))

        for n in range(img_sample_n):
            for c in range(img_channel):
                img = Image.fromarray(delta_arr[n, c])
                img = img.resize(self.__target_shape)
                resized_delta_arr = np.asarray(img)
                result_arr[n, c] = resized_delta_arr

        result_arr = super().back_propagate(result_arr)

        for n in range(img_sample_n):
            for c in range(img_channel):
                img = Image.fromarray(result_arr[n, c])
                img = img.resize(self.__target_shape)
                resized_delta_arr = np.asarray(img)
                _delta_arr[n, c] = resized_delta_arr

        return _delta_arr

    def get_target_shape(self):
        ''' getter '''
        return self.__target_shape

    def set_target_shape(self, value):
        ''' setter '''
        self.__target_shape = value
    
    target_shape = property(get_target_shape, set_target_shape)
