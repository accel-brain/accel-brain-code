# -*- coding: utf-8 -*-
import numpy as np
from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
from pygan.discriminativemodel.cnn_model import CNNModel


class SeqCNNModel(CNNModel):
    '''
    Convolutional Neural Network as a Discriminator.

    This model observes sequencal data as image-like data.

    If the length of sequence is `T` and the dimension is `D`, 
    image-like matrix will be configured as a `T` × `D` matrix.
    '''

    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
        '''
        return super().inference(
            # Add rank for channel.
            np.expand_dims(observed_arr, axis=1)
        )

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
        return delta_arr[:, 0]
