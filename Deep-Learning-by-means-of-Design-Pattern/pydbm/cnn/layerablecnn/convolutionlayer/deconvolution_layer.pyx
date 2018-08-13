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
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef int img_sample_n = img_arr.shape[0]

        cdef int n = 0
        cdef int c = 0
        cdef np.ndarray[DOUBLE_t, ndim=2] arr = self.__transpose(img_arr[n, c])
        cdef int t_height = arr.shape[0]
        cdef int t_width = arr.shape[1]
        
        if kernel_height != t_height or kernel_width != t_width:
            raise ValueError("The shape of kernel is invalid. Infered shape is " + str((t_height, t_width)))

        cdef np.ndarray[DOUBLE_t, ndim=4] _img_arr = np.zeros((
            img_sample_n,
            channel,
            t_height,
            t_width
        ))

        for n in range(img_sample_n):
            for c in range(channel):
                _img_arr[n, c] = self.__transpose(img_arr[n, c])

        return super().forward_propagate(_img_arr)

    def __transpose(self, np.ndarray[DOUBLE_t, ndim=2] arr):
        '''
        Transpose of convolving `arr`.
        
        Args:
            arr:        `np.ndarray` of image array.
        
        Returns:
            Transposed array.
        '''
        
        for _ in range(self.__t_pad_width):
            arr = np.insert(arr, obj=list(range(arr.shape[0])), values=0, axis=0)
            arr = np.insert(arr, obj=list(range(arr.shape[1])), values=0, axis=1)
        return arr

    def get_t_pad_width(self):
        ''' getter '''
        return self.__t_pad_width

    def set_t_pad_width(self, value):
        ''' setter '''
        self.__t_pad_width = value
    
    t_pad_width = property(get_t_pad_width, set_t_pad_width)
