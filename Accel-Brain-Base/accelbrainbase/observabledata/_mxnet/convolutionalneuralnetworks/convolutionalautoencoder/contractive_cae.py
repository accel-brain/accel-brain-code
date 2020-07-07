# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class ContractiveCAE(ConvolutionalAutoEncoder):
    '''
    Convolutional Contracitve Auto-Encoder.

    A stack of Convolutional Auto-Encoder (Masci, J., et al., 2011) 
    forms a convolutional neural network(CNN), which are among the most successful models 
    for supervised image classification.  Each Convolutional Auto-Encoder is trained 
    using conventional on-line gradient descent without additional regularization terms.
    
    In this library, Convolutional Auto-Encoder is also based on Encoder/Decoder scheme.
    The encoder is to the decoder what the Convolution is to the Deconvolution.
    The Deconvolution also called transposed convolutions 
    "work by swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    The First-Order Contractive Auto-Encoder(Rifai, S., et al., 2011) executes 
    the representation learning by adding a penalty term to the classical 
    reconstruction cost function. This penalty term corresponds to 
    the Frobenius norm of the Jacobian matrix of the encoder activations
    with respect to the input and results in a localized space 
    contraction which in turn yields robust features on the activation layer.

    Analogically, the Contractive Convolutional Auto-Encoder calculates the penalty term.
    But it differs in that the operation of the deconvolution intervenes insted of inner product.

    References:
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Masci, J., Meier, U., Cireşan, D., & Schmidhuber, J. (2011, June). Stacked convolutional auto-encoders for hierarchical feature extraction. In International Conference on Artificial Neural Networks (pp. 52-59). Springer, Berlin, Heidelberg.
        - Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011, June). Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of the 28th International Conference on International Conference on Machine Learning (pp. 833-840). Omnipress.
        - Rifai, S., Mesnil, G., Vincent, P., Muller, X., Bengio, Y., Dauphin, Y., & Glorot, X. (2011, September). Higher order contractive auto-encoder. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645-660). Springer, Berlin, Heidelberg.
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
        penalty_arr = self.feature_points_arr * (1 - self.feature_points_arr)

        if self.penalty_lambda > 0:
            penalty_arr = self.decoder.forward_propagation(F, penalty_arr)
        
        return decoded_arr + (self.penalty_lambda * penalty_arr)

    def get_penalty_lambda(self):
        ''' getter for lambda. '''
        return self.__penalty_lambda
    
    def set_penalty_lambda(self, value):
        ''' setter for lambda. '''
        self.__penalty_lambda = value
    
    penalty_lambda = property(get_penalty_lambda, set_penalty_lambda)
