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
    # `Tuple` of target shape. The shape is (`height`, `width`).
    __target_shape = (10, 10)
    # Number of values padded to the edges of each axis.
    __t_pad_width = 0

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
        cdef int sample_n = self.graph.weight_arr.shape[0]
        cdef int channel = self.graph.weight_arr.shape[1]
        cdef int kernel_height = self.graph.weight_arr.shape[2]
        cdef int kernel_width = self.graph.weight_arr.shape[3]

        for _ in range(self.__t_pad_width):
            img_arr = np.insert(img_arr, obj=list(range(1, img_arr.shape[2] - 1)), values=0, axis=2)
            img_arr = np.insert(img_arr, obj=list(range(1, img_arr.shape[3] - 1)), values=0, axis=3)

        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]

        cdef int result_h = self.__target_shape[0]
        cdef int result_w = self.__target_shape[1]

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

        self.img_arr = img_arr
        self.reshaped_img_arr = reshaped_img_arr
        self.reshaped_weight_arr = reshaped_weight_arr

        return self.graph.activation_function.activate(_result_arr)

    def back_propagate(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN layers.
        
        Override.
        
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

        for _ in range(self.__t_pad_width):
            delta_arr = np.insert(delta_arr, obj=list(range(1, delta_arr.shape[2] - 1)), values=0, axis=2)
            delta_arr = np.insert(delta_arr, obj=list(range(1, delta_arr.shape[3] - 1)), values=0, axis=3)

        cdef int img_sample_n = delta_arr.shape[0]
        cdef int img_channel = delta_arr.shape[1]
        cdef int img_height = delta_arr.shape[2]
        cdef int img_width = delta_arr.shape[3]

        delta_arr = delta_arr.transpose(0, 2, 3, 1)

        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr = delta_arr.reshape(-1, img_sample_n)
        cdef np.ndarray[DOUBLE_t, ndim=1] delta_bias_arr = _delta_arr.sum(axis=0)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr = np.dot(self.reshaped_img_arr.T, _delta_arr)
        delta_weight_arr = delta_weight_arr.transpose(1, 0)
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_weight_arr = delta_weight_arr.reshape(
            sample_n,
            channel,
            kernel_height,
            kernel_width
        )

        if self.delta_bias_arr is None:
            self.delta_bias_arr = delta_bias_arr
        else:
            self.delta_bias_arr += delta_bias_arr

        if self.delta_weight_arr is None:
            self.delta_weight_arr = _delta_weight_arr
        else:
            self.delta_weight_arr += _delta_weight_arr

        cdef np.ndarray[DOUBLE_t, ndim=2] delta_reshaped_img_arr = np.dot(_delta_arr, self.reshaped_weight_arr.T)
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_img_arr = self.affine_to_img(
            delta_reshaped_img_arr,
            self.img_arr, 
            kernel_height, 
            kernel_width, 
            self.__stride, 
            self.__pad
        )
        return delta_img_arr

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
            target_shape:       `Tuple` of target shape. The shape is (`height`, `width`).
        
        Returns:
            2-rank image array.
        '''
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = img_arr.shape[2]
        cdef int img_width = img_arr.shape[3]
        
        self.__img_height = img_height
        self.__img_width = img_width

        cdef int result_height = self.__target_shape[0]
        cdef int result_width = self.__target_shape[1]

        cdef int h_d = (result_height - img_height) // 2
        cdef int w_d = (result_width - img_width) // 2

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

        if h_d > 0:
            obj_list = list(range(h_d))
            for _ in range(kernel_height):
                [obj_list.append(v) for v in list(range(pad_arr.shape[2] - h_d, pad_arr.shape[2]))]
            pad_arr = np.insert(pad_arr, obj=obj_list, values=0, axis=2)

        if w_d > 0:
            obj_list = list(range(w_d))
            for _ in range(kernel_width):
                [obj_list.append(v) for v in list(range(pad_arr.shape[3] - w_d, pad_arr.shape[3]))]
            pad_arr = np.insert(pad_arr, obj=obj_list, values=0, axis=3)

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
                result_arr[:, :, height, width, :, :] = pad_arr[:, :, height+h_d:max_height+h_d:stride, width+w_d:max_width+w_d:stride]

        result_arr = result_arr.transpose(0, 4, 5, 1, 2, 3)
        cdef np.ndarray[DOUBLE_t, ndim=2] _result_arr = result_arr.reshape(
            img_sample_n * result_height * result_width,
            -1
        )
        return _result_arr

    def affine_to_img(
        self,
        np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr,
        np.ndarray[DOUBLE_t, ndim=4] img_arr, 
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
        cdef int img_sample_n = img_arr.shape[0]
        cdef int img_channel = img_arr.shape[1]
        cdef int img_height = self.__img_height
        cdef int img_width = self.__img_width

        cdef int result_height = self.__target_shape[0]
        cdef int result_width = self.__target_shape[1]

        cdef int h_d = (result_height - img_height) // 2
        cdef int w_d = (result_width - img_width) // 2

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
        if h_d > 0:
            obj_list = list(range(h_d))
            for _ in range(kernel_height):
                [obj_list.append(v) for v in list(range(result_arr.shape[2] - h_d, result_arr.shape[2]))]
            result_arr = np.insert(result_arr, obj=obj_list, values=0, axis=2)

        if w_d > 0:
            obj_list = list(range(w_d))
            for _ in range(kernel_width):
                [obj_list.append(v) for v in list(range(result_arr.shape[3] - w_d, result_arr.shape[3]))]
            result_arr = np.insert(result_arr, obj=obj_list, values=0, axis=3)

        cdef int height
        cdef int width
        cdef int max_height = 0
        cdef int max_width = 0
        for height in range(kernel_height):
            max_height = height + stride * result_height
            for width in range(kernel_width):
                max_width = width + stride * result_width
                result_arr[:, :, height+h_d:max_height+h_d:stride, width+w_d:max_width+w_d:stride] += _reshaped_img_arr[:, :, height, width, :, :]

        return result_arr[:, :, pad:img_height + pad, pad:img_width + pad]

    def get_target_shape(self):
        ''' getter '''
        return self.__target_shape

    def set_target_shape(self, value):
        ''' setter '''
        self.__target_shape = value
    
    target_shape = property(get_target_shape, set_target_shape)
    
    def get_t_pad_width(self):
        ''' getter '''
        return self.__t_pad_width

    def set_t_pad_width(self, value):
        ''' setter '''
        self.__t_pad_width = value
    
    t_pad_width = property(get_t_pad_width, set_t_pad_width)
