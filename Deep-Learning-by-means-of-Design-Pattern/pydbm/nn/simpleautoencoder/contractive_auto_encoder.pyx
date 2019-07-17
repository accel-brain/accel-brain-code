# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.nn.simple_auto_encoder import SimpleAutoEncoder
from pydbm.activation.logistic_function import LogisticFunction
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ContractiveAutoEncoder(SimpleAutoEncoder):
    '''
    Contractive Auto-Encoder.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011, June). Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of the 28th International Conference on International Conference on Machine Learning (pp. 833-840). Omnipress.
        - Rifai, S., Mesnil, G., Vincent, P., Muller, X., Bengio, Y., Dauphin, Y., & Glorot, X. (2011, September). Higher order contractive auto-encoder. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645-660). Springer, Berlin, Heidelberg.
    '''

    # Positive hyperparameter that controls the strength of the regularization.
    __penalty_lambda = 1.0

    def __init__(
        self,
        encoder,
        decoder,
        computable_loss,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        pre_learned_path_tuple=None
    ):
        '''
        Init.
        
        Args:
            encoder:                        is-a `NeuralNetwork`.
            decoder:                        is-a `NeuralNetwork`.
            computable_loss:                Loss function.
            verificatable_result:           Verification function.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            tol:                            Tolerance for the optimization.
            tld:                            Tolerance for deviation of loss.
        '''
        if isinstance(encoder.nn_layer_list[-1].graph.activation_function, LogisticFunction) is False:
            raise TypeError("The type of `activation_function` in last layer must be `LogisticFunction`.")

        super().__init__(
            encoder,
            decoder,
            computable_loss,
            verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            tol=tol,
            tld=tld,
            pre_learned_path_tuple=pre_learned_path_tuple
        )

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN.
        
        Args:
            observed_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef np.ndarray encoded_arr = self.encoder.inference(observed_arr)
        cdef np.ndarray decoded_arr = self.decoder.inference(encoded_arr)

        cdef np.ndarray feature_points_arr = encoded_arr * (1 - encoded_arr)
        nn_layer_list = self.encoder.nn_layer_list[::-1]
        for i in range(len(nn_layer_list)):
            feature_points_arr = np.dot(feature_points_arr, nn_layer_list[i].graph.weight_arr.T)

        feature_points_arr = (feature_points_arr - feature_points_arr.mean()) / (feature_points_arr.std() + 1e-08)
        self.computable_loss.penalty_delta_arr = feature_points_arr * self.penalty_lambda
        self.computable_loss.penalty_term = np.nanmean(feature_points_arr) * self.penalty_lambda

        return decoded_arr

    def get_penalty_lambda(self):
        '''
        getter for Positive hyperparameter 
        that controls the strength of the regularization.
        '''
        return self.__penalty_lambda
    
    def set_penalty_lambda(self, value):
        '''
        setter for Positive hyperparameter 
        that controls the strength of the regularization.
        '''
        if isinstance(value, float) is False:
            raise TypeError("The type of `penalty_lambda` must be `float`.")
        if value <= 0:
            raise ValueError("The value of `penalty_lambda` must be more than `0.0`.")
        self.__penalty_lambda = value
    
    penalty_lambda = property(get_penalty_lambda, set_penalty_lambda)
