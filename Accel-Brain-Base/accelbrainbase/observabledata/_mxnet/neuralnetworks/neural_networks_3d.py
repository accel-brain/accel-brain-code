# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
import mxnet as mx


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

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
                    The shape is ...
                    - batch.
                    - sequence.
                    - dimention.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        x = F.reshape(x, shape=(self.batch_size * self.seq_len, -1))
        for i in range(len(self.activation_list)):
            x = self.fc_list[i](x)
            if self.activation_list[i] == "identity_adjusted":
                x = x / F.sum(F.ones_like(x))
            elif self.activation_list[i] != "identity":
                x = F.Activation(x, self.activation_list[i])
            if self.dropout_forward_list[i] is not None:
                x = self.dropout_forward_list[i](x)

        x = F.reshape(x, shape=(self.batch_size, self.seq_len, -1))

        return x
