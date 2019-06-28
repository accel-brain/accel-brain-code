# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.clustering.interface.auto_encodable import AutoEncodable
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder


class ConvolutionalDEC(AutoEncodable):
    '''
    The Deep Embedded Clustering(DEC) with Convolutional Neural Networks.

    References:
        - Guo, X., Liu, X., Zhu, E., & Yin, J. (2017, November). Deep clustering with convolutional autoencoders. In International Conference on Neural Information Processing (pp. 373-382). Springer, Cham.
        - Guo, X., Gao, L., Liu, X., & Yin, J. (2017, June). Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI (pp. 1753-1759).
        - Xie, J., Girshick, R., & Farhadi, A. (2016, June). Unsupervised deep embedding for clustering analysis. In International conference on machine learning (pp. 478-487).
    '''
    # is-a `ConvolutionalAutoEncoder`.
    __convolutional_auto_encoder = None

    def get_auto_encoder_model(self):
        ''' getter '''
        return self.__convolutional_auto_encoder
    
    def set_auto_encoder_model(self, value):
        ''' setter '''
        if isinstance(value, ConvolutionalAutoEncoder) is False:
            raise TypeError("The type of `auto_encoder_model` must be `ConvolutionalAutoEncoder`.")

        self.__convolutional_auto_encoder = value

    auto_encoder_model = property(get_auto_encoder_model, set_auto_encoder_model)

    def get_inferencing_mode(self):
        ''' getter '''
        return self.__convolutional_auto_encoder.opt_params.inferencing_mode
    
    def set_inferencing_mode(self, value):
        ''' setter '''
        self.__convolutional_auto_encoder.opt_params.inferencing_mode = value

    inferencing_mode = property(get_inferencing_mode, set_inferencing_mode)

    def pre_learn(self, np.ndarray observed_arr, np.ndarray target_arr=None):
        '''
        Pre-learning.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
            target_arr:         `np.ndarray` of noised observed data points.
        '''
        self.__convolutional_auto_encoder.learn(observed_arr, target_arr)

    def inference(self, np.ndarray observed_arr):
        '''
        Inferencing.

        Args:
            observed_arr:       `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced data.
        '''
        return self.__convolutional_auto_encoder.inference(observed_arr)

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

    def backward_auto_encoder(self, np.ndarray delta_arr, encoder_only_flag=True):
        '''
        Pass down to the Auto-Encoder as backward.

        Args:
            delta_arr:              `np.ndarray` of delta.
            encoder_only_flag:      Pass down to encoder only or decoder/encoder.

        Returns:
            `np.ndarray` of delta.
        '''
        if encoder_only_flag is True:
            layerable_cnn_list = self.__convolutional_auto_encoder.layerable_cnn_list[::-1]
            for i in range(len(layerable_cnn_list)):
                delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
                delta_arr = layerable_cnn_list[i].graph.deactivation_function.forward(delta_arr)

            return delta_arr
        else:
            return self.__convolutional_auto_encoder.back_propagation(delta_arr)

    def optimize_auto_encoder(self, learning_rate, epoch, encoder_only_flag=True):
        '''
        Optimize Auto-Encoder.

        Args:
            learning_rate:          Learning rate.
            epoch:                  Now epoch.
            encoder_only_flag:      Optimize encoder only or decoder/encoder.
        '''
        # Convotlutional Auto-Encoder operates deconvolution as a transposed convolution. 
        if encoder_only_flag is True or encoder_only_flag is False:
            self.__convolutional_auto_encoder.optimize(learning_rate, epoch)
