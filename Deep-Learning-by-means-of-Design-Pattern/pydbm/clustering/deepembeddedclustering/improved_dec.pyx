# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.clustering.deep_embedded_clustering import DeepEmbeddedClustering


class ImprovedDEC(DeepEmbeddedClustering):
    '''
    The Improved Deep Embedded Clustering(iDEC).

    References:
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    
    # Gamma.
    __gamma = 0.5

    def get_gamma(self):
        ''' getter for Gamma. '''
        return self.__gamma
    
    def set_gamma(self, value):
        ''' setter for Gamma. '''
        self.__gamma = value

    gamma = property(get_gamma, set_gamma)

    # MSE
    __mean_squared_error = MeanSquaredError()

    def forward_propagation(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points and do soft assignment.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of result of soft assignment.
        '''
        self.__observed_arr = observed_arr
        self.__pred_arr = self.auto_encodable.inference(observed_arr)
        return super().forward_propagation(observed_arr)
    
    def compute_loss(self, p_arr, q_arr):
        '''
        Compute loss.

        Args:
            p_arr:      `np.ndarray` of result of soft assignment.
            q_arr:      `np.ndarray` of target distribution.
        
        Returns:
            (loss, `np.ndarray` of delta)
        '''
        loss = super().compute_loss(p_arr, q_arr)
        reconstructed_loss = self.__mean_squared_error.compute_loss(self.__pred_arr, self.__observed_arr)
        self.__delta_arr = self.__mean_squared_error.compute_delta(self.__pred_arr, self.__observed_arr)
        return loss + (reconstructed_loss * self.__gamma)

    def back_propagation(self):
        '''
        Back propagation.

        Returns:
            `np.ndarray` of delta.
        '''
        super().back_propagation()
        self.auto_encodable.backward_auto_encoder(
            self.__delta_arr,
            encoder_only_flag=False
        )

    def optimize(self, learning_rate, epoch):
        '''
        Optimize.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        params_list = [
            self.mu_arr
        ]
        grads_list = [
            self.delta_mu_arr
        ]
        params_list = self.opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        self.mu_arr = params_list[0]
        self.auto_encodable.optimize_auto_encoder(learning_rate, epoch, encoder_only_flag=False)
