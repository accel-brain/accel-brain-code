# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.nn.neural_network import NeuralNetwork
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class SimpleAutoEncoder(object):
    '''
    Auto-Encoder.

    References:
        - Kamyshanska, H., & Memisevic, R. (2014). The potential energy of an autoencoder. IEEE transactions on pattern analysis and machine intelligence, 37(6), 1261-1273.
    '''
    # is-a `ComputableLoss`.
    __computable_loss = None

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
        if isinstance(encoder, NeuralNetwork) is False:
            raise TypeError("The type of `encoder` must be `NeuralNetwork`.")
        if isinstance(decoder, NeuralNetwork) is False:
            raise TypeError("The type of `decoder` must be `NeuralNetwork`.")

        self.__encoder = encoder
        self.__decoder = decoder

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss`.")

        if isinstance(verificatable_result, VerificatableResult):
            self.__verificatable_result = verificatable_result
        else:
            raise TypeError("The type of `verificatable_result` must be `VerificatableResult`.")

        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__test_size_rate = test_size_rate
        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []

        logger = getLogger("pydbm")
        self.__logger = logger
        
        self.__logger.debug("Setup NN layers and the parameters.")

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=2] observed_arr,
        np.ndarray target_arr=None
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of labeled data.
                            If `None`, the function of this NN model is equivalent to Convolutional Auto-Encoder.

        '''
        self.__logger.debug("NN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = 0
        if target_arr is not None:
            row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=2] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=2] batch_observed_arr
        cdef np.ndarray batch_target_arr

        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round(self.__test_size_rate * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
            train_target_arr = target_arr[train_index]
            test_target_arr = target_arr[test_index]
        else:
            train_observed_arr = observed_arr
            train_target_arr = observed_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=2] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_arr
        
        best_encoder_weight_params_list = []
        best_encoder_bias_params_list = []
        best_decoder_weight_params_list = []
        best_decoder_bias_params_list = []

        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.__encoder.opt_params.inferencing_mode = False
                self.__decoder.opt_params.inferencing_mode = False

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    loss = self.__computable_loss.compute_loss(
                        pred_arr,
                        batch_target_arr
                    )
                    loss = loss + train_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            best_encoder_weight_params_list, 
                            best_encoder_bias_params_list,
                            best_decoder_weight_params_list, 
                            best_decoder_bias_params_list
                        )
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                        loss = self.__computable_loss.compute_loss(
                            pred_arr,
                            batch_target_arr
                        )
                        loss = loss + train_weight_decay

                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr,
                        batch_target_arr
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_encoder_weight_params_list = []
                        best_encoder_bias_params_list = []
                        best_decoder_weight_params_list = []
                        best_decoder_bias_params_list = []

                        for i in range(len(self.__encoder.nn_layer_list)):
                            best_encoder_weight_params_list.append(self.__encoder.nn_layer_list[i].graph.weight_arr)
                            best_encoder_bias_params_list.append(self.__encoder.nn_layer_list[i].graph.bias_arr)
                        for i in range(len(self.__decoder.nn_layer_list)):
                            best_decoder_weight_params_list.append(self.__decoder.nn_layer_list[i].graph.weight_arr)
                            best_decoder_bias_params_list.append(self.__decoder.nn_layer_list[i].graph.bias_arr)

                        self.__logger.debug("Best params are updated.")

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        raise

                if self.__test_size_rate > 0:
                    self.__encoder.opt_params.inferencing_mode = True
                    self.__decoder.opt_params.inferencing_mode = True

                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr,
                        test_batch_target_arr
                    )
                    test_loss = test_loss + test_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(
                            best_encoder_weight_params_list, 
                            best_encoder_bias_params_list,
                            best_decoder_weight_params_list, 
                            best_decoder_bias_params_list
                        )
                        # Re-try
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_weight_decay,
                                test_penalty=test_weight_decay
                            )

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(
            best_encoder_weight_params_list, 
            best_encoder_bias_params_list,
            best_decoder_weight_params_list, 
            best_decoder_bias_params_list
        )
        self.__logger.debug("end. ")

    def __remember_best_params(
        self, 
        best_encoder_weight_params_list, 
        best_encoder_bias_params_list,
        best_decoder_weight_params_list, 
        best_decoder_bias_params_list
    ):
        '''
        Remember best parameters.
        
        Args:
            best_encoder_weight_params_list:    `list` of weight parameters in encoder.
            best_encoder_bias_params_list:      `list` of bias parameters in encoder.
            best_decoder_weight_params_list:    `list` of weight parameters in decoder.
            best_decoder_bias_params_list:      `list` of bias parameters in decoder.

        '''
        if len(best_encoder_weight_params_list) and len(best_encoder_bias_params_list):
            for i in range(len(self.__encoder.nn_layer_list)):
                self.__encoder.nn_layer_list[i].graph.weight_arr = best_encoder_weight_params_list[i]
                self.__encoder.nn_layer_list[i].graph.bias_arr = best_encoder_bias_params_list[i]
            self.__logger.debug("Encoder's best params are saved.")
        if len(best_decoder_weight_params_list) and len(best_decoder_bias_params_list):
            for i in range(len(self.__decoder.nn_layer_list)):
                self.__decoder.nn_layer_list[i].graph.weight_arr = best_decoder_weight_params_list[i]
                self.__decoder.nn_layer_list[i].graph.bias_arr = best_decoder_bias_params_list[i]
            self.__logger.debug("Decoder's best params are saved.")

    def inference(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        cdef np.ndarray pred_arr = self.forward_propagation(
            observed_arr
        )
        return pred_arr

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=2] observed_arr):
        '''
        Forward propagation in NN.
        
        Args:
            observed_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef np.ndarray encoded_arr = self.__encoder.inference(observed_arr)
        cdef np.ndarray decoded_arr = self.__decoder.inference(encoded_arr)
        return decoded_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation in NN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        delta_arr = self.__decoder.back_propagation(delta_arr)
        delta_arr = self.__encoder.back_propagation(delta_arr)
        return delta_arr

    def optimize(self, double learning_rate, int epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        self.__encoder.optimize(learning_rate, epoch)
        self.__decoder.optimize(learning_rate, epoch)

    def get_verificatable_result(self):
        ''' getter '''
        if isinstance(self.__verificatable_result, VerificatableResult):
            return self.__verificatable_result
        else:
            raise TypeError()

    def set_verificatable_result(self, value):
        ''' setter '''
        if isinstance(value, VerificatableResult):
            self.__verificatable_result = value
        else:
            raise TypeError()
    
    verificatable_result = property(get_verificatable_result, set_verificatable_result)

    def get_computable_loss(self):
        ''' getter '''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter '''
        self.__computable_loss = value

    computable_loss = property(get_computable_loss, set_computable_loss)

    def get_encoder(self):
        ''' getter '''
        return self.__encoder
    
    def set_encoder(self, value):
        ''' setter '''
        if isinstance(value, NeuralNetwork):
            raise TypeError("The type of `encoder` must be `NeuralNetwork`.")
        self.__encoder = value
    
    encoder = property(get_encoder, set_encoder)

    def get_decoder(self):
        ''' getter '''
        return self.__decoder
    
    def set_decoder(self, value):
        ''' setter '''
        if isinstance(value, NeuralNetwork):
            raise TypeError("The type of `decoder` must be `NeuralNetwork`.")
        self.__decoder = value
    
    decoder = property(get_decoder, set_decoder)
