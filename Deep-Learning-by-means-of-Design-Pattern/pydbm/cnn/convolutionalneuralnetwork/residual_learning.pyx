# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class ResidualLearning(ConvolutionalNeuralNetwork):
    '''
    Residual Learning Framework.
    '''
    # Learning or not. If `True`, this class executes residual learning.
    __learn_mode = True

    def __init__(
        self,
        layerable_cnn_list,
        int epochs,
        int batch_size,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        computable_loss,
        opt_params,
        verificatable_result,
        double test_size_rate=0.3,
        tol=1e-15,
        save_flag=False
    ):
        '''
        Init.
        
        Override.
        
        Args:
            layerable_cnn_list:     The `list` of `LayerableCNN`.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.
            tol:                            Tolerance for the optimization.
            save_flag:                      If `True`, save `np.ndarray` of inferenced test data in training.

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
            save_flag
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
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            observed_arr:   `np.ndarray` of labeled data.
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
                self.__logger.debug("Input shape in CNN layer: " + str(i + 1))
                self.__logger.debug((
                    img_arr.shape[0],
                    img_arr.shape[1],
                    img_arr.shape[2],
                    img_arr.shape[3]
                ))
                img_arr = self.layerable_cnn_list[i].forward_propagate(img_arr)
            except:
                self.__logger.debug("Error raised in CNN layer " + str(i + 1))
                raise

        self.__logger.debug("-" * 100)
        self.__logger.debug("Propagated shape in CNN layer: " + str(i + 1))
        self.__logger.debug((
            img_arr.shape[0],
            img_arr.shape[1],
            img_arr.shape[2],
            img_arr.shape[3]
        ))
        self.__logger.debug("-" * 100)
        if self.__learn_mode is True:
            img_arr = img_arr + _img_arr
            self.__logger.debug("Residual learning...")

        return img_arr
