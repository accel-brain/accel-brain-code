# -*- coding: utf-8 -*-
import numpy as np
from pygan.discriminativemodel.cnn_model import CNNModel


class SeqCNNModel(CNNModel):
    '''
    Convolutional Neural Network as a Discriminator.

    This model observes sequencal data as image-like data.

    If the length of sequence is `T` and the dimension is `D`, 
    image-like matrix will be configured as a `T` × `D` matrix.
    '''

    # Add channel or not.
    __add_channel_flag = False

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        if observed_arr.ndim < 4:
            # Add rank for channel.
            observed_arr = np.expand_dims(observed_arr, axis=1)
            self.__add_channel_flag = True
        else:
            self.__add_channel_flag = False

        return super().inference(observed_arr)

    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        delta_arr = super().learn(grad_arr, fix_opt_flag)
        if self.__add_channel_flag is True:
            return delta_arr[:, 0]
        else:
            return delta_arr

    def feature_matching_forward(self, observed_arr):
        '''
        Forward propagation in only first or intermediate layer
        for so-called Feature matching.

        Args:
            observed_arr:       `np.ndarray` of observed data points.

        Returns:
            `np.ndarray` of outputs.
        '''
        if observed_arr.ndim < 4:
            # Add rank for channel.
            observed_arr = np.expand_dims(observed_arr, axis=1)

        return super().feature_matching_forward(observed_arr)
