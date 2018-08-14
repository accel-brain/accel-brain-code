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

    def __init__(self, graph):
        '''
        Init.
        
        Args:
            graph:      is-a `Synapse`.
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
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]

        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        for _ in range(self.__t_pad_width):
            img_arr = np.insert(img_arr, obj=list(range(img_arr.shape[2])), values=0, axis=2)
            img_arr = np.insert(img_arr, obj=list(range(img_arr.shape[3])), values=0, axis=3)

        return super().forward_propagate(img_arr)

    def get_t_pad_width(self):
        ''' getter '''
        return self.__t_pad_width

    def set_t_pad_width(self, value):
        ''' setter '''
        self.__t_pad_width = value
    
    t_pad_width = property(get_t_pad_width, set_t_pad_width)
