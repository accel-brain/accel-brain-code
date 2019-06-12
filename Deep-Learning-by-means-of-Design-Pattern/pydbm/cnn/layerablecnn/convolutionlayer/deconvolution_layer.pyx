# -*- coding: utf-8 -*-
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class DeconvolutionLayer(ConvolutionLayer):
    '''
    Deconvolution Layer.

    Deconvolution also called transposed convolutions
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

    '''

    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            img_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=4] result_arr = self.deconvolve(img_arr)
        cdef np.ndarray[DOUBLE_t, ndim=2] _result_arr = result_arr.reshape((
            result_arr.shape[0],
            -1
        ))
        result_arr = _result_arr.reshape((
            result_arr.shape[0],
            result_arr.shape[1],
            result_arr.shape[2],
            result_arr.shape[3]
        ))
        if self.graph.bias_arr is not None:
            result_arr += self.graph.bias_arr.reshape((
                1,
                result_arr.shape[1],
                result_arr.shape[2],
                result_arr.shape[3]
            ))

        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr = self.affine_to_matrix(
            img_arr,
            kernel_height, 
            kernel_width, 
            self.graph.stride, 
            self.graph.pad
        )
        self.__reshaped_img_arr = reshaped_img_arr
        self.__channel = img_arr.shape[1]
        return self.graph.activation_function.activate(result_arr)

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.

        Args:
            delta_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        delta_arr = self.graph.activation_function.derivative(delta_arr)

        cdef int sample_n = self.graph.weight_arr.shape[0]
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef int img_sample_n = delta_arr.shape[0]
        cdef int img_channel = delta_arr.shape[1]
        cdef int img_height = delta_arr.shape[2]
        cdef int img_width = delta_arr.shape[3]

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.convolve(delta_arr, no_bias_flag=True)
        delta_arr = delta_arr.transpose(0, 2, 3, 1)
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = delta_arr.reshape(-1, sample_n)
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_bias_arr = delta_arr.sum(axis=0)
        _delta_arr = delta_arr.reshape(-1, channel)
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr = self.__reshaped_img_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr = np.dot(reshaped_img_arr.T, _delta_arr)
        delta_weight_arr = delta_weight_arr.transpose(1, 0)
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_weight_arr = delta_weight_arr.reshape(
            sample_n,
            channel,
            kernel_height,
            kernel_width
        )
        if self.graph.bias_arr is None:
            self.graph.bias_arr = np.zeros((1, img_channel * img_height * img_width))

        if self.delta_bias_arr is None:
            self.delta_bias_arr = delta_bias_arr.reshape(1, -1)
        else:
            self.delta_bias_arr += delta_bias_arr.reshape(1, -1)

        if self.delta_weight_arr is None:
            self.delta_weight_arr = _delta_weight_arr
        else:
            self.delta_weight_arr += _delta_weight_arr

        return delta_img_arr
