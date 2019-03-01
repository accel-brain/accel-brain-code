# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class LayerableCNN(metaclass=ABCMeta):
    '''
    The abstract class of convolutional neural network.
    '''

    @abstractproperty
    def graph(self):
        ''' Graph which is-a `Synapse`. '''
        raise NotImplementedError()

    @abstractproperty
    def delta_weight_arr(self):
        ''' Delta of weight matirx.'''
        raise NotImplementedError()
        
    @abstractproperty
    def delta_bias_arr(self):
        ''' Delta of bias vector.'''
        raise NotImplementedError()

    @abstractmethod
    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.

        Args:
            matriimg_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        raise NotImplementedError()

    @abstractmethod
    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.

        Args:
            delta_arr:      4-rank array like or sparse matrix.
        
        Returns:
            3-rank array like or sparse matrix.
        '''
        raise NotImplementedError()

    def affine_to_matrix(
        self,
        np.ndarray[DOUBLE_t, ndim=4] img_arr,
        int kernel_height,
        int kernel_width,
        int stride,
        int pad
    ):
        '''
        Affine transform for Convolution.
        
        Args:
            img_arr:            `np.ndarray` of 4-rank image array.
            kernel_height:      Height of kernel.
            kernel_width:       Width of kernel.
            stride:             Stride.
            pad:                padding value.
        
        Returns:
            2-rank image array.
        '''
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]

        cdef int result_height = int((img_height + 2 * pad - kernel_height) // stride) + 1
        cdef int result_width = int((img_width + 2 * pad - kernel_width) // stride) + 1

        cdef np.ndarray[DOUBLE_t, ndim=4] pad_arr = np.pad(
            img_arr,
            [
                (0, 0),
                (0, 0),
                (pad, pad),
                (pad, pad)
            ],
            'constant'
        )

        cdef np.ndarray[DOUBLE_t, ndim=6] result_arr = np.zeros(
            (
                img_sample_n,
                img_channel,
                kernel_height, 
                kernel_width, 
                result_height, 
                result_width
            )
        )

        cdef int max_height = 0
        cdef int max_width = 0
        for height in range(kernel_height):
            max_height = height + stride * result_height
            for width in range(kernel_width):
                max_width = width + stride * result_width
                result_arr[:, :, height, width, :, :] = pad_arr[:, :, height:max_height:stride, width:max_width:stride]

        result_arr = result_arr.transpose(0, 4, 5, 1, 2, 3)
        cdef np.ndarray[DOUBLE_t, ndim=2] _result_arr = result_arr.reshape(
            img_sample_n * result_height * result_width,
            -1
        )
        return _result_arr

    def affine_to_img(
        self,
        np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr,
        int img_sample_n,
        int img_channel,
        int img_height,
        int img_width, 
        int kernel_height, 
        int kernel_width, 
        int stride, 
        int pad
    ):
        '''
        Affine transform for Convolution.
        
        Args:
            reshaped_img_arr:   `np.ndarray` of 2-rank image array.
            img_arr:            `np.ndarray` of 4-rank image array.
            kernel_height:      Height of kernel.
            kernel_width:       Width of kernel.
            stride:             Stride.
            pad:                padding value.
        
        Returns:
            2-rank image array.
        '''
        img_height = stride * (img_height - 1) + kernel_height - (2 * pad)
        img_width = stride * (img_width - 1) + kernel_width - (2 * pad)
        cdef int result_height = int((img_height + 2 * pad - kernel_height) // stride) + 1
        cdef int result_width = int((img_width + 2 * pad - kernel_width) // stride) + 1
        cdef np.ndarray[DOUBLE_t, ndim=6] _reshaped_img_arr = reshaped_img_arr.reshape(
            (
                img_sample_n,
                result_height,
                result_width,
                img_channel,
                kernel_height,
                kernel_width
            )
        )
        _reshaped_img_arr = _reshaped_img_arr.transpose(0, 3, 4, 5, 1, 2)

        cdef np.ndarray[DOUBLE_t, ndim=4] result_arr = np.zeros(
            (
                img_sample_n, 
                img_channel,
                img_height + 2 * pad + stride - 1,
                img_width + 2 * pad + stride - 1
            )
        )
        
        cdef int height
        cdef int width
        cdef int max_height = 0
        cdef int max_width = 0
        for height in range(kernel_height):
            max_height = height + stride * result_height
            for width in range(kernel_width):
                max_width = width + stride * result_width
                result_arr[:, :, height:max_height:stride, width:max_width:stride] += _reshaped_img_arr[:, :, height, width, :, :]

        return result_arr[:, :, pad:img_height + pad, pad:img_width + pad]

    def reset_delta(self):
        '''
        Reset delta.
        '''
        self.delta_weight_arr = None
        self.delta_bias_arr = None
