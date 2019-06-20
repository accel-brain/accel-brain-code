# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.clustering.deep_embedded_clustering import DeepEmbeddedClustering
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder


class SimpleDEC(DeepEmbeddedClustering):
    '''
    The Deep Embedded Clustering(DEC).

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''

    # is-a `SimpleAutoEncoer`.
    __simple_auto_encoder = None

    def get_auto_encoder_model(self):
        ''' getter '''
        return self.__simple_auto_encoder
    
    def set_auto_encoder_model(self, value):
        ''' setter '''
        if isinstance(value, SimpleAutoEncoder) is False:
            raise TypeError("The type of `auto_encoder_model` must be `SimpleAutoEncoder`.")

        self.__simple_auto_encoder = value

    auto_encoder_model = property(get_auto_encoder_model, set_auto_encoder_model)

    def get_inferencing_mode(self):
        ''' getter '''
        return self.__simple_auto_encoder.encoder.opt_params.inferencing_mode
    
    def set_inferencing_mode(self, value):
        ''' setter '''
        self.__simple_auto_encoder.encoder.opt_params.inferencing_mode = value
        self.__simple_auto_encoder.decoder.opt_params.inferencing_mode = value

    inferencing_mode = property(get_inferencing_mode, set_inferencing_mode)

    def pre_learn(self, np.ndarray observed_arr):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            feature_generator:  is-a `FeatureGenerator`.
        '''
        self.__simple_auto_encoder.learn(observed_arr)

    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        return self.__simple_auto_encoder.encoder.inference(observed_arr)

    def backward_auto_encoder(self, np.ndarray delta_arr):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:      `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        return self.__simple_auto_encoder.encoder.back_propagation(delta_arr)

    def optimize_auto_encoder(self, learning_rate, epoch):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        self.__simple_auto_encoder.encoder.optimize(learning_rate, epoch)
