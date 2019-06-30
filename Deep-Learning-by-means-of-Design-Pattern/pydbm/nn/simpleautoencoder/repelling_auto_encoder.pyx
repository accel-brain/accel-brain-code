# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder
from pydbm.activation.logistic_function import LogisticFunction
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class RepellingAutoEncoder(SimpleAutoEncoder):
    '''
    Repelling Auto-Encoder.

    References:
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN.
        
        Args:
            observed_arr:    `np.ndarray` of observed data points.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef np.ndarray encoded_arr = self.encoder.inference(observed_arr)
        cdef np.ndarray decoded_arr = self.decoder.inference(encoded_arr)

        cdef np.ndarray feature_points_arr = encoded_arr.reshape((encoded_arr.shape[0], -1))
        feature_points_arr = (feature_points_arr - feature_points_arr.mean()) / (feature_points_arr.std() + 1e-08)

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

        return decoded_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation in NN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        delta_arr = self.decoder.back_propagation(delta_arr)
        delta_arr = delta_arr + self.__penalty_delta_arr.reshape(delta_arr.copy().shape)
        delta_arr = self.encoder.back_propagation(delta_arr)
        return delta_arr
