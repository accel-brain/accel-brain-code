# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class RepellingCAE(ConvolutionalAutoEncoder):
    '''
    Repelling Convolutional Auto-Encoder.

    A stack of Convolutional Auto-Encoder (Masci, J., et al., 2011) 
    forms a convolutional neural network(CNN), which are among the most successful models 
    for supervised image classification.  Each Convolutional Auto-Encoder is trained 
    using conventional on-line gradient descent without additional regularization terms.
    
    In this library, Convolutional Auto-Encoder is also based on Encoder/Decoder scheme.
    The encoder is to the decoder what the Convolution is to the Deconvolution.
    The Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    This Convolutional Auto-Encoder calculates the Repelling regularizer(Zhao, J., et al., 2016) as a penalty term.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    # penalty lambda.
    __penalty_lambda = 1e-05

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        decoded_arr = super().forward_propagation(F, x)
        penalty_arr = self.feature_points_arr

        row_arr = F.ones_like(penalty_arr)
        N = F.mean(F.sum(row_arr, axis=1))

        repelling = 0.0
        if self.penalty_lambda > 0:
            penalty_arr = F.flatten(penalty_arr)
            for i in range(self.batch_size):
                for j in range(self.batch_size):
                    if i == j:
                        continue
                    repelling = repelling + F.dot(
                        penalty_arr[i].T,
                        penalty_arr[j]
                    ) / (F.sqrt(
                        F.dot(
                            penalty_arr[i],
                            penalty_arr[i]
                        )) * F.sqrt(
                            F.dot(
                                penalty_arr[j],
                                penalty_arr[j]
                            )
                        )
                    )

            repelling = repelling / (N * (N - 1))
            repelling_arr = repelling * self.feature_points_arr
            penalty_arr = self.decoder.forward_propagation(F, repelling_arr)

        return decoded_arr + (self.penalty_lambda * penalty_arr)

    def get_penalty_lambda(self):
        ''' getter for lambda.'''
        return self.__penalty_lambda
    
    def set_penalty_lambda(self, value):
        ''' setter for lambda.'''
        self.__penalty_lambda = value
    
    penalty_lambda = property(get_penalty_lambda, set_penalty_lambda)
