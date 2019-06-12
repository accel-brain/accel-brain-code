# -*- coding: utf-8 -*-
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.synapse_list import Synapse
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ConvolutionLayer(LayerableCNN):
    '''
    Convolution Layer.
    
    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.

    '''
    # Computation graph which is-a `Synapse`.
    __graph = None
    # Delta of weight matrix.
    __delta_weight_arr = np.array([[]])
    # Delta of bias vector.
    __delta_bias_arr = np.array([[]])
    # Height of image.
    __img_height = None
    # Width of image.
    __img_width = None
    # Reshaped image matrix.
    __reshaped_img_arr = None

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

        self.__stride = graph.stride
        self.__pad = graph.pad
        
        self.__delta_weight_arr = None
        self.__delta_bias_arr = None
        self.__reshaped_img_arr = None

    def forward_propagate(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN layers.
        
        Override.

        Args:
            img_arr:      4-rank array like or sparse matrix.
        
        Returns:
            4-rank array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=4] result_arr = self.convolve(img_arr)
        return self.graph.activation_function.activate(result_arr)

    def convolve(self, np.ndarray[DOUBLE_t, ndim=4] img_arr, no_bias_flag=False):
        '''
        Convolution.
        
        Args:
            img_arr:        4-rank array like or sparse matrix.
            no_bias_flag:   Use bias or not.
        
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

        cdef int propagated_height = int((img_height + 2 * self.__pad - kernel_height) // self.__stride) + 1
        cdef int propagated_width = int((img_width + 2 * self.__pad - kernel_width) // self.__stride) + 1

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
        )
        cdef np.ndarray[DOUBLE_t, ndim=4] _result_arr = result_arr.reshape(
            sample_n, 
            propagated_height, 
            propagated_width, 
            -1
        )
        _result_arr = _result_arr.transpose(0, 3, 1, 2)
        if no_bias_flag is False and self.graph.bias_arr is not None:
            _result_arr += self.graph.bias_arr.reshape((
                1,
                _result_arr.shape[1],
                _result_arr.shape[2],
                _result_arr.shape[3]
            ))

        self.__img_arr = img_arr
        self.__propagated_height = propagated_height
        self.__propagated_width = propagated_width
        self.__img_height = img_height
        self.__img_width = img_width
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

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.deconvolve(delta_arr)

        delta_arr = delta_arr.transpose(0, 2, 3, 1)
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = delta_arr.reshape(-1, sample_n)
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_bias_arr = delta_arr.sum(axis=0)
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

        if self.__delta_bias_arr is None:
            self.__delta_bias_arr = delta_bias_arr.reshape(1, -1)
        else:
            self.__delta_bias_arr += delta_bias_arr.reshape(1, -1)

        if self.__delta_weight_arr is None:
            self.__delta_weight_arr = _delta_weight_arr
        else:
            self.__delta_weight_arr += _delta_weight_arr

        return delta_img_arr

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
        cdef int sample_n = self.graph.weight_arr.shape[0]
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        cdef int img_sample_n = delta_arr.shape[0]
        cdef int img_channel = delta_arr.shape[1]
        cdef int img_height = delta_arr.shape[2]
        cdef int img_width = delta_arr.shape[3]

        delta_arr = delta_arr.transpose(0, 2, 3, 1)
        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = delta_arr.reshape(-1, img_sample_n)
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_weight_arr = self.graph.weight_arr.reshape(sample_n, -1).T
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_reshaped_img_arr = np.dot(_delta_arr, reshaped_weight_arr.T)

        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.affine_to_img(
            delta_reshaped_img_arr,
            img_sample_n,
            channel,
            img_height,
            img_width, 
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

    def get_img_arr(self):
        ''' getter '''
        return self.__img_arr

    def set_img_arr(self, value):
        ''' setter '''
        self.__img_arr = value

    img_arr = property(get_img_arr, set_img_arr)

    def get_reshaped_img_arr(self):
        ''' getter '''
        return self.__reshaped_img_arr

    def set_reshaped_img_arr(self, value):
        ''' setter '''
        self.__reshaped_img_arr = value
    
    reshaped_img_arr = property(get_reshaped_img_arr, set_reshaped_img_arr)
    
    def get_reshaped_weight_arr(self):
        ''' getter '''
        return self.__reshaped_weight_arr

    def set_reshaped_weight_arr(self, value):
        ''' setter '''
        self.__reshaped_weight_arr = value
    
    reshaped_weight_arr = property(get_reshaped_weight_arr, set_reshaped_weight_arr)

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
