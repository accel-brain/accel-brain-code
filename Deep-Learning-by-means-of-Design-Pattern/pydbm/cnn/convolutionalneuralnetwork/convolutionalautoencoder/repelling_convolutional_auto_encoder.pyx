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
        cdef np.ndarray pt_arr = np.zeros(s ** 2)
        k = 0
        for i in range(s):
            for j in range(s):
                if i == j:
                    continue
                pt_arr[k] = np.dot(feature_points_arr[i].T, feature_points_arr[j]) / (np.sqrt(np.dot(feature_points_arr[i], feature_points_arr[i])) * np.sqrt(np.dot(feature_points_arr[j], feature_points_arr[j])))
                k += 1

        self.computable_loss.penalty_arr = pt_arr.sum() / (N * (N - 1))
        return result_arr
