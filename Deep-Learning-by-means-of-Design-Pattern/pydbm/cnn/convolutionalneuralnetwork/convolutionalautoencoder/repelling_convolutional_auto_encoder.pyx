# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class RepellingConvolutionalAutoEncoder(ConvolutionalAutoEncoder):
    '''
    Repelling Convolutional Auto-Encoder which is-a `ConvolutionalNeuralNetwork`.

    This Convolutional Auto-Encoder calculates the Repelling regularizer(Zhao, J., et al., 2016)
    as a penalty term.

    **Note** that it is only an *intuitive* application in this library.

    References:
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef np.ndarray result_arr = super().forward_propagation(img_arr)

        cdef np.ndarray feature_points_arr = self.extract_feature_points_arr()
        feature_points_arr = feature_points_arr.reshape((feature_points_arr.shape[0], -1))
        cdef int N = feature_points_arr.shape[1]
        cdef int s = feature_points_arr.shape[0]
        cdef np.ndarray[DOUBLE_t, ndim=1] pt_arr = np.zeros(s ** 2)
        k = 0
        for i in range(s):
            for j in range(s):
                if i == j:
                    continue
                pt_arr[k] = np.dot(feature_points_arr[i].T, feature_points_arr[j]) / (np.sqrt(np.dot(feature_points_arr[i], feature_points_arr[i])) * np.sqrt(np.dot(feature_points_arr[j], feature_points_arr[j])))
                k += 1

        self.computable_loss.penalty_term = pt_arr.sum() / (N * (N - 1))

        cdef np.ndarray[DOUBLE_t, ndim=2] penalty_delta_arr = np.dot(
            self.computable_loss.penalty_term, 
            feature_points_arr
        )
        self.__penalty_delta_arr = penalty_delta_arr
        return result_arr

    def back_propagation(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN.
        
        Override.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        cdef int i = 0
        cdef int sample_n = delta_arr.shape[0]
        cdef int kernel_height
        cdef int kernel_width
        cdef int img_sample_n
        cdef int img_channel
        cdef int img_height
        cdef int img_width

        cdef np.ndarray[DOUBLE_t, ndim=2] _delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] delta_bias_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] reshaped_img_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_weight_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] _delta_weight_arr

        for i in range(len(self.layerable_cnn_list)):
            img_sample_n = delta_arr.shape[0]
            img_channel = delta_arr.shape[1]
            img_height = delta_arr.shape[2]
            img_width = delta_arr.shape[3]

            if self.layerable_cnn_list[i].graph.constant_flag is False:
                kernel_height = self.layerable_cnn_list[i].graph.weight_arr.shape[2]
                kernel_width = self.layerable_cnn_list[i].graph.weight_arr.shape[3]
                reshaped_img_arr = self.layerable_cnn_list[i].affine_to_matrix(
                    delta_arr,
                    kernel_height, 
                    kernel_width, 
                    self.layerable_cnn_list[i].graph.stride, 
                    self.layerable_cnn_list[i].graph.pad
                )
                delta_bias_arr = delta_arr.sum(axis=0)
            delta_arr = self.layerable_cnn_list[i].convolve(delta_arr, no_bias_flag=True)
            channel = delta_arr.shape[1]

            if self.layerable_cnn_list[i].graph.constant_flag is False:
                _delta_arr = delta_arr.reshape(-1, sample_n)
                delta_weight_arr = np.dot(reshaped_img_arr.T, _delta_arr)

                delta_weight_arr = delta_weight_arr.transpose(1, 0)
                _delta_weight_arr = delta_weight_arr.reshape(
                    sample_n,
                    kernel_height,
                    kernel_width,
                    -1
                )
                _delta_weight_arr = _delta_weight_arr.transpose((0, 3, 1, 2))

                if self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr is None:
                    self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr = delta_bias_arr.reshape(1, -1)
                else:
                    self.layerable_cnn_list[i].graph.delta_deconvolved_bias_arr += delta_bias_arr.reshape(1, -1)

                if self.layerable_cnn_list[i].graph.deconvolved_bias_arr is None:
                    self.layerable_cnn_list[i].graph.deconvolved_bias_arr = np.zeros((
                        1, 
                        img_channel * img_height * img_width
                    ))

                if self.layerable_cnn_list[i].delta_weight_arr is None:
                    self.layerable_cnn_list[i].delta_weight_arr = _delta_weight_arr
                else:
                    self.layerable_cnn_list[i].delta_weight_arr += _delta_weight_arr

        delta_arr = delta_arr + self.__penalty_delta_arr.reshape((delta_arr.copy().shape))

        cdef np.ndarray[DOUBLE_t, ndim=2] hidden_activity_arr
        if self.opt_params.dropout_rate > 0:
            hidden_activity_arr = delta_arr.reshape((delta_arr.shape[0], -1))
            hidden_activity_arr = self.opt_params.de_dropout(hidden_activity_arr)
            delta_arr = hidden_activity_arr.reshape((
                delta_arr.shape[0],
                delta_arr.shape[1],
                delta_arr.shape[2],
                delta_arr.shape[3]
            ))

        layerable_cnn_list = self.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
            delta_arr = layerable_cnn_list[i].graph.deactivation_function.forward(delta_arr)

        return delta_arr
