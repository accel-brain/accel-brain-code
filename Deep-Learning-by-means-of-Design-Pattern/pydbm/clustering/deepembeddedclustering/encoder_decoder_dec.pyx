# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.clustering.deep_embedded_clustering import DeepEmbeddedClustering
from pydbm.rnn.encoder_decoder_controller import EncoderDecoderController


class EncoderDecoderDec(DeepEmbeddedClustering):
    '''
    The Deep Embedded Clustering(DEC) with Encoder/Decoder based on LSTM.

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).

    '''

    # is-a `EncoderDecoderController`.
    __encoder_decoder_controller = None

    def get_auto_encoder_model(self):
        ''' getter '''
        return self.__encoder_decoder_controller
    
    def set_auto_encoder_model(self, value):
        ''' setter '''
        if isinstance(value, EncoderDecoderController) is False:
            raise TypeError("The type of `auto_encoder_model` must be `EncoderDecoderController`.")

        self.__encoder_decoder_controller = value

    auto_encoder_model = property(get_auto_encoder_model, set_auto_encoder_model)

    def pre_learn(self, np.ndarray observed_arr):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            feature_generator:  is-a `FeatureGenerator`.
        '''
        self.__encoder_decoder_controller.learn(observed_arr)

    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        return self.__encoder_decoder_controller.encoder.inference(observed_arr)

    def backward_auto_encoder(self, np.ndarray delta_arr):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:      `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        return self.__encoder_decoder_controller.encoder.back_propagation(delta_arr)

    def optimize_auto_encoder(self, learning_rate, epoch):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        self.__encoder_decoder_controller.encoder.optimize(learning_rate, epoch)
