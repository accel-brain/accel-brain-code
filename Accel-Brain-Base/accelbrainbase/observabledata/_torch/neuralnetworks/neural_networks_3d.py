# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
import torch
import torch.nn as nn
import torch.nn.functional as F


class NN3DHybrid(NeuralNetworks):
    '''
    3D Neural networks.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
    '''
    # Batch size.
    __batch_size = None
    # The length of series.
    __seq_len = None

    def get_batch_size(self):
        ''' getter '''
        return self.__batch_size
    
    def set_batch_size(self, value):
        ''' setter '''
        self.__batch_size = value

    batch_size = property(get_batch_size, set_batch_size)

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value
    
    seq_len = property(get_seq_len, set_seq_len)

    def forward(self, x):
        '''
        Forward with torch.

        Args:
            x:      `tensor` of observed data points.
        
        Returns:
            `tensor` of inferenced feature points.
        '''
        x = torch.reshape(
            x, 
            shape=(
                self.batch_size * self.seq_len, 
                -1
            )
        )
        x = super().forward(x)
        x = torch.reshape(
            x, 
            shape=(
                self.batch_size, 
                self.seq_len, 
                -1
            )
        )

        return x
