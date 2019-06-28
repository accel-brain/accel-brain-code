# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.clustering.interface.auto_encodable import AutoEncodable
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder


class SimpleDEC(AutoEncodable):
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

    def pre_learn(self, np.ndarray observed_arr, np.ndarray target_arr=None):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            target_arr:         `np.ndarray` of noised observed data points.
        '''
        self.__simple_auto_encoder.learn(observed_arr, target_arr)

    def inference(self, np.ndarray observed_arr):
        '''
        Inferencing.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced data.
        '''
        return self.__simple_auto_encoder.inference(observed_arr)

    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        return self.__simple_auto_encoder.encoder.inference(observed_arr)

    def backward_auto_encoder(self, np.ndarray delta_arr, encoder_only_flag=True):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:          `np.ndarray` of delta.
            encoder_only_flag:  Pass down to encoder only or decoder/encoder.
        
        Returns:
            `np.ndarray` of delta.
        '''
        if encoder_only_flag is True:
            return self.__simple_auto_encoder.encoder.back_propagation(delta_arr)
        else:
            return self.__simple_auto_encoder.back_propagation(delta_arr)

    def optimize_auto_encoder(self, learning_rate, epoch, encoder_only_flag=True):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:          Learning rate.
            epoch:                  Now epoch.
            encoder_only_flag:      Optimize encoder only or decoder/encoder.
        '''
        if encoder_only_flag is True:
            self.__simple_auto_encoder.encoder.optimize(learning_rate, epoch)
        else:
            self.__simple_auto_encoder.optimize(learning_rate, epoch)
