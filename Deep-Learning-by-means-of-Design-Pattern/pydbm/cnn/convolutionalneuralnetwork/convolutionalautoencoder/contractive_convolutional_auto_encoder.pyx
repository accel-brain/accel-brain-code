# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.convolutionalneuralnetwork.convolutional_auto_encoder import ConvolutionalAutoEncoder
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer
from pydbm.activation.logistic_function import LogisticFunction
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ContractiveConvolutionalAutoEncoder(ConvolutionalAutoEncoder):
    '''
    Contractive Convolutional Auto-Encoder which is-a `ConvolutionalNeuralNetwork`.

    The First-Order Contractive Auto-Encoder(Rifai, S., et al., 2011) executes 
    the representation learning by adding a penalty term to the classical 
    reconstruction cost function. This penalty term corresponds to 
    the Frobenius norm of the Jacobian matrix of the encoder activations
    with respect to the input and results in a localized space 
    contraction which in turn yields robust features on the activation layer. 

    Analogically, the Contractive Convolutional Auto-Encoder calculates the penalty term.
    But it differs in that the operation of the deconvolution intervenes insted of inner product.

    **Note** that it is only an *intuitive* application in this library.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
        - Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011, June). Contractive auto-encoders: Explicit invariance during feature extraction. In Proceedings of the 28th International Conference on International Conference on Machine Learning (pp. 833-840). Omnipress.
        - Rifai, S., Mesnil, G., Vincent, P., Muller, X., Bengio, Y., Dauphin, Y., & Glorot, X. (2011, September). Higher order contractive auto-encoder. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 645-660). Springer, Berlin, Heidelberg.
    '''

    # Positive hyperparameter that controls the strength of the regularization.
    __penalty_lambda = 1.0

    def __init__(
        self,
        layerable_cnn_list,
        computable_loss,
        opt_params,
        verificatable_result,
        int epochs=100,
        int batch_size=100,
        double learning_rate=1e-05,
        double learning_attenuate_rate=0.1,
        int attenuate_epoch=50,
        double test_size_rate=0.3,
        tol=1e-15,
        tld=100.0,
        save_flag=False,
        pre_learned_path_list=None
    ):
        '''
        Init.
        
        Override.
        
        Args:
            layerable_cnn_list:             The `list` of `ConvolutionLayer`.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
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
            save_flag:                      If `True`, save `np.ndarray` of inferenced test data in training.
            pre_learned_path_list:          `list` of file path that stores pre-learned parameters.

        Exceptions:
            TypeError:                      When the type of `activation_function` in last layer is not `LogisticFunction`.
                                            Only the case of a sigmoid nonlinearity is theoretically acceptable 
                                            for computational considerations(Rifai, S., et al., 2011).
        '''
        if isinstance(layerable_cnn_list[-1].graph.activation_function, LogisticFunction) is False:
            raise TypeError("The type of `activation_function` in last layer must be `LogisticFunction`.")

        super().__init__(
            layerable_cnn_list=layerable_cnn_list,
            computable_loss=computable_loss,
            opt_params=opt_params,
            verificatable_result=verificatable_result,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            test_size_rate=test_size_rate,
            tol=tol,
            tld=tld,
            save_flag=save_flag,
            pre_learned_path_list=pre_learned_path_list
        )

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef np.ndarray result_arr = super().forward_propagation(img_arr)

        cdef np.ndarray feature_points_arr = self.extract_feature_points_arr()
        feature_points_arr = feature_points_arr * (1 - feature_points_arr)
        layerable_cnn_list = self.layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            feature_points_arr = layerable_cnn_list[i].graph.activation_function.backward(
                feature_points_arr
            )
            feature_points_arr = layerable_cnn_list[i].deconvolve(feature_points_arr)

        self.computable_loss.penalty_delta_arr = feature_points_arr * self.penalty_lambda
        self.computable_loss.penalty_term = np.nanmean(feature_points_arr) * self.penalty_lambda

        return result_arr

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
