# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.clustering.deep_embedded_clustering import DeepEmbeddedClustering
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder


class ConvolutionalDEC(DeepEmbeddedClustering):
    '''
    The Deep Embedded Clustering(DEC) with Convolutional Neural Networks.

    References:
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''

    def __init__(self, convolutional_auto_encoder):
        '''
        Init.

        Args:
            convolutional_auto_encoder:        is-a `SimpleAutoEncoder`.
        '''
        if isinstance(convolutional_auto_encoder, ConvolutionalAutoEncoder) is False:
            raise TypeError("The type of `convolutional_auto_encoder` must be `ConvolutionalAutoEncoder`.")

        self.__convolutional_auto_encoder = convolutional_auto_encoder

    def pre_learn(self, np.ndarray observed_arr):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            feature_generator:  is-a `FeatureGenerator`.
        '''
        self.__convolutional_auto_encoder.learn(observed_arr)

    def embed_feature_points(self, np.ndarray observed_arr):
        '''
        Embed and extract feature points.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of feature points.
        '''
        _ = self.__convolutional_auto_encoder.inference(observed_arr)
        return self.__convolutional_auto_encoder.extract_feature_points_arr()

    def backward_auto_encoder(self, np.ndarray delta_arr):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:      `np.ndarray` of delta.
        
        Returns:
            `np.ndarray` of delta.
        '''
        layerable_cnn_list = self.__convolutional_auto_encoder.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
            delta_arr = layerable_cnn_list[i].graph.deactivation_function.forward(delta_arr)

        return delta_arr

    def optimize_auto_encoder(self, learning_rate, epoch):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:      Learning rate.
            epoch:              Now epoch.
        '''
        self.__convolutional_auto_encoder.encoder.optimize(learning_rate, epoch)
