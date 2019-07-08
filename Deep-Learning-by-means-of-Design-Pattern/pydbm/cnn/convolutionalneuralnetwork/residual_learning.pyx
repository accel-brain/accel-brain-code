# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ResidualLearning(ConvolutionalNeuralNetwork):
    '''
    Deep Residual Learning Framework which hypothesize that
    "it is easier to optimize the residual mapping than to optimize the original, 
    unreferenced mapping. To the extreme, if an identity mapping were optimal, 
    it would be easier to push the residual to zero than to fit an identity mapping 
    by a stack of nonlinear layers." (He, K. et al., 2016, p771.)
    
    References:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

    '''

    # Learning or not. If `True`, this class executes residual learning.
    __learn_mode = True

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
        tld=10000.0,
        save_flag=False,
        pre_learned_path_list=None
    ):
        '''
        Init.
        
        Override.
        
        Args:
            layerable_cnn_list:             The `list` of `LayerableCNN`.
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

        '''
        super().__init__(
            layerable_cnn_list,
            epochs,
            batch_size,
            learning_rate,
            learning_attenuate_rate,
            attenuate_epoch,
            computable_loss,
            opt_params,
            verificatable_result,
            test_size_rate,
            tol,
            tld,
            save_flag,
            pre_learned_path_list
        )

        logger = getLogger("pydbm")
        self.__logger = logger
        self.__learn_mode = True
        self.__logger.debug("Setup CNN layers and the parameters.")

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=4] observed_arr,
        np.ndarray target_arr=None
    ):
        '''
        Learn with Deep Residual Learning Framework.
        
        "With the residual learning reformulation, if identity mappings are optimal, 
        the solvers may simply drive the weights of the multiple nonlinear layers
        toward zero to approach identity mappings". (He, K. et al., 2016, p771.)
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of labeled data.
                            If `None`, the function of this cnn model is equivalent to Convolutional Auto-Encoder.

        References:
            - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
        
        '''
        self.__learn_mode = True
        super().learn(observed_arr, target_arr)

    def inference(self, np.ndarray[DOUBLE_t, ndim=4] observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        self.__learn_mode = False
        return super().inference(observed_arr)

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=4] img_arr):
        '''
        Forward propagation in CNN.
        
        Override.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef int i = 0
        self.__logger.debug("-" * 100)
        cdef np.ndarray[DOUBLE_t, ndim=4] _img_arr
        if self.__learn_mode is True:
            _img_arr = img_arr.copy()

        for i in range(len(self.layerable_cnn_list)):
            try:
                img_arr = self.layerable_cnn_list[i].forward_propagate(img_arr)
            except:
                self.__logger.debug("Error raised in CNN layer " + str(i + 1))
                raise

            if self.layerable_cnn_list[i].graph.constant_flag is False:
                self.weight_decay_term += self.opt_params.compute_weight_decay(
                    self.layerable_cnn_list[i].graph.weight_arr
                )

        if self.__learn_mode is True:
            img_arr = img_arr + _img_arr
            self.__logger.debug("Residual learning...")

            if self.opt_params.dropout_rate > 0:
                hidden_activity_arr = img_arr.reshape((img_arr.shape[0], -1))
                hidden_activity_arr = self.opt_params.dropout(hidden_activity_arr)
                img_arr = hidden_activity_arr.reshape((
                    img_arr.shape[0],
                    img_arr.shape[1],
                    img_arr.shape[2],
                    img_arr.shape[3]
                ))

        return img_arr
