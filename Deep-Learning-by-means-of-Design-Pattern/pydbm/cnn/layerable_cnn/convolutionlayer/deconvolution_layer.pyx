# -*- coding: utf-8 -*-
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
        Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

    '''
    # Number of values padded to the edges of each axis.
    __t_pad_width = 1

    def __init__(self, graph, int stride=1, int pad=0):
        '''
        Init.
        
        Args:
            graph:      is-a `Synapse`.
            stride:     Stride.
            pad:        Padding.
        '''
        self.__stride = stride
        self.__pad = pad
        
        super().__init__(graph, stride, pad)
    
    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            img_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int n, c

        for n in range(img_sample_n):
            for c in range(img_channel):
                img_arr[n, c] = self.__transpose(img_arr[n, c])

        return super().forward_propagate(img_arr)

    def __transpose(self, arr):
        '''
        Transpose of convolving `arr`.
        
        Args:
            arr:        `np.ndarray` of image array.
        
        Returns:
            Transposed array.
        '''
        output_height = arr.shape[0] * (2 + self.__t_pad_width - 1) + 1
        output_width = arr.shape[1] * (2 + self.__t_pad_width - 1) + 1
        pad_arr = np.zeros((output_height, output_width))
        pad_list = []
        for height in range(arr.shape[0]):
            w_list = []
            for width in range(arr.shape[1]):
                w_list.append(
                    np.pad(
                        arr[height, width].reshape(-1, 1),
                        pad_width=(self.__t_pad_width, 0),
                        mode="constant",
                    )
                )
                w_arr = np.hstack(w_list)
            pad_list.append(w_arr)
        pad_arr[:output_height-1, :output_width-1] = np.vstack(pad_list)
        return pad_arr

    def get_t_pad_width(self):
        ''' getter '''
        return self.__t_pad_width

    def set_t_pad_width(self, value):
        ''' setter '''
        self.__t_pad_width = value
    
    t_pad_width = property(get_t_pad_width, set_t_pad_width)
