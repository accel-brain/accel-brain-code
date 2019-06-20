# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
ctypedef np.float64_t DOUBLE_t


class AutoEncodable(metaclass=ABCMeta):
    '''
    The interface of the Deep Embedded Clustering(DEC).

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''

    @abstractproperty
    def auto_encoder_model(self):
        ''' Model object of Auto-Encoder. '''
        raise NotImplementedError()

    @abstractproperty
    def inferencing_mode(self):
        ''' `inferencing_mode` for `auto_encoder_model`. '''
        raise NotImplementedError()

    @abstractmethod
    def pre_learn(self, np.ndarray observed_arr):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        raise NotImplementedError()

    @abstractmethod
    def backward_auto_encoder(self, np.ndarray delta_arr):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:      `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        raise NotImplementedError()

    @abstractmethod
    def optimize_auto_encoder(self, learning_rate, epoch):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        raise NotImplementedError()
