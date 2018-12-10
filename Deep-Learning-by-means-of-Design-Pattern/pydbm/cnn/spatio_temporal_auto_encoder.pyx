# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class SpatioTemporalAutoEncoder(object):
    '''
    Spatio-Temporal Auto-Encoder.
    '''
    
    def __init__(
        self,
        layerable_cnn_list,
        encoder,
        decoder,
        int epochs,
        int batch_size,
        double learning_rate,
        double learning_attenuate_rate,
        int attenuate_epoch,
        computable_loss,
        opt_params,
        verificatable_result,
        double test_size_rate=0.3,
        int fully_connected_dim=100,
        fully_connected_activation=None,
        tol=1e-15,
        tld=100.0,
        save_flag=False,
        pre_learned_dir=None
    ):
        '''
        Init.
        
        Args:
            layerable_cnn_list:             The `list` of `LayerableCNN`.
            encoder:                        is-a `ReconstructableModel`.
            decoder:                        is-a `ReconstructableModel`.
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            fully_connected_dim:            Dimension of fully-connected layer between Convolution layer and Encoder/Decoder.
            fully_connected_activation:     is-a `ActivatingFunctionInterface`.
            computable_loss:                Loss function.
            opt_params:                     Optimization function.
            verificatable_result:           Verification function.
            tol:                            Tolerance for the optimization.
            tld:                            Tolerance for deviation of loss.
            save_flag:                      If `True`, save `np.ndarray` of inferenced test data in training.
            pre_learned_dir:                Path to directory that stores pre-learned parameters.

        '''
        if isinstance(encoder, ReconstructableModel) is False:
            raise TypeError()
        if isinstance(decoder, ReconstructableModel) is False:
            raise TypeError()

        for layerable_cnn in layerable_cnn_list:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError("The type of value of `layerable_cnn` must be `LayerableCNN`.")

        if pre_learned_dir is not None:
            for i in range(len(layerable_cnn_list)):
                layerable_cnn_list[i].graph.save_pre_learned_params(pre_learned_dir + "spatio_cnn_" + str(i) + ".npz")
            encoder.graph.load_pre_learned_params(pre_learned_dir + "temporal_encoder.npz")
            decoder.graph.load_pre_learned_params(pre_learned_dir + "temporal_decoder.npz")

        self.__layerable_cnn_list = layerable_cnn_list
        self.__encoder = encoder
        self.__decoder = decoder

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError("The type of `computable_loss` must be `ComputableLoss`.")

        if isinstance(opt_params, OptParams):
            self.__opt_params = opt_params
            self.__dropout_rate = self.__opt_params.dropout_rate
        else:
            raise TypeError("The type of `opt_params` must be `OptParams`.")

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
        self.__fully_connected_dim = fully_connected_dim
        self.__fully_connected_weight_arr = None
        
        if fully_connected_activation is not None:
            if isinstance(fully_connected_activation, ActivatingFunctionInterface) is False:
                raise TypeError()
            self.__fully_connected_activation = fully_connected_activation
        else:
            self.__fully_connected_activation = None

        self.__tol = tol
        self.__tld = tld

        self.__memory_tuple_list = []
        
        self.__save_flag = save_flag
        
        if pre_learned_dir is None:
            self.__learn_flag = True
        else:
            self.__learn_flag = False

        logger = getLogger("pydbm")
        self.__logger = logger
        
        self.__logger.debug("Setup CNN layers and the parameters.")

    def learn(
        self,
        np.ndarray[DOUBLE_t, ndim=5] observed_arr,
        np.ndarray target_arr=None
    ):
        '''
        Learn.
        
        Args:
            observed_arr:   `np.ndarray` of observed data points.
            target_arr:     `np.ndarray` of labeled data.
                            If `None`, the function of this cnn model is equivalent to Convolutional Auto-Encoder.

        '''
        self.__logger.debug("CNN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = 0
        if target_arr is not None:
            row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=5] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=5] batch_observed_arr
        cdef np.ndarray batch_target_arr

        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()
        else:
            if target_arr.ndim == 2:
                target_arr = target_arr.reshape((target_arr.shape[0], 1, target_arr.shape[1]))

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
        cdef np.ndarray[DOUBLE_t, ndim=5] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_arr
        
        best_weight_params_list = []
        best_bias_params_list = []
        self.__encoder_best_params_list = []
        self.__decoder_best_params_list = []
        self.__temporal_min_loss = None

        self.__learn_flag = True
        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            for epoch in range(self.__epochs):
                self.__opt_params.dropout_rate = self.__dropout_rate

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate
                self.__now_epoch = epoch
                self.__now_learning_rate = learning_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]

                try:
                    self.__learn_flag = True
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    loss = self.__computable_loss.compute_loss(
                        pred_arr[:, -1],
                        batch_target_arr[:, -1]
                    )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        loss = self.__computable_loss.compute_loss(
                            pred_arr[:, -1],
                            batch_target_arr[:, -1]
                        )
                        
                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr[:, -1],
                        batch_target_arr[:, -1]
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.__layerable_cnn_list)):
                            best_weight_params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
                        self.__logger.debug("Convolutional Auto-Encoder's best params are updated.")

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
                    self.__learn_flag = False
                    self.__opt_params.dropout_rate = 0.0
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )

                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr[:, -1],
                        test_batch_target_arr[:, -1]
                    )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.__save_flag is True:
                        np.save("test_pred_arr_" + str(epoch), test_pred_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Convolutional Auto-Encoder's loss.")
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr[:, -1], 
                                train_label_arr=batch_target_arr[:, -1],
                                test_pred_arr=test_pred_arr[:, -1],
                                test_label_arr=test_batch_target_arr[:, -1]
                            )
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Encoder/Decoder's loss: ")
                            self.__logger.debug("Training: " + str(self.__encoder_decoder_loss) + " Test: " + str(self.__test_encoder_decoder_loss))
                            self.__logger.debug("-" * 100)

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
        self.__logger.debug("end. ")

    def learn_generated(self, feature_generator):
        '''
        Learn features generated by `FeatureGenerator`.
        
        Args:
            feature_generator:    is-a `FeatureGenerator`.

        '''
        if isinstance(feature_generator, FeatureGenerator) is False:
            raise TypeError("The type of `feature_generator` must be `FeatureGenerator`.")

        self.__logger.debug("CNN starts learning.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=5] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=5] batch_observed_arr
        cdef np.ndarray batch_target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray[DOUBLE_t, ndim=5] pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=5] test_pred_arr
        cdef np.ndarray[DOUBLE_t, ndim=4] delta_arr

        best_weight_params_list = []
        best_bias_params_list = []
        self.__encoder_best_params_list = []
        self.__decoder_best_params_list = []
        self.__temporal_min_loss = None

        self.__learn_flag = True
        try:
            self.__memory_tuple_list = []
            loss_list = []
            min_loss = None
            eary_stop_flag = False
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1
                self.__opt_params.dropout_rate = self.__dropout_rate

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate / self.__learning_attenuate_rate
                self.__now_epoch = epoch
                self.__now_learning_rate = learning_rate
                try:
                    self.__learn_flag = True
                    pred_arr = self.inference(batch_observed_arr)
                    ver_pred_arr = pred_arr.copy()
                    loss = self.__computable_loss.compute_loss(
                        pred_arr[:, -1],
                        batch_target_arr[:, -1]
                    )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        self.__logger.debug("Re-try.")
                        pred_arr = self.inference(batch_observed_arr)
                        ver_pred_arr = pred_arr.copy()
                        loss = self.__computable_loss.compute_loss(
                            pred_arr[:, -1],
                            batch_target_arr[:, -1]
                        )

                    delta_arr = self.__computable_loss.compute_delta(
                        pred_arr[:, -1],
                        batch_target_arr[:, -1]
                    )
                    delta_arr = self.back_propagation(delta_arr)
                    self.optimize(learning_rate, epoch)
                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        best_weight_params_list = []
                        best_bias_params_list = []

                        for i in range(len(self.__layerable_cnn_list)):
                            best_weight_params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
                            best_bias_params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
                        self.__logger.debug("Convolutional Auto-Encoder's best params are updated.")

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
                    self.__opt_params.dropout_rate = 0.0
                    self.__learn_flag = False
                    test_pred_arr = self.forward_propagation(
                        test_batch_observed_arr
                    )
                    test_loss = self.__computable_loss.compute_loss(
                        test_pred_arr[:, -1],
                        test_batch_target_arr[:, -1]
                    )

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
                        # Re-try.
                        self.__logger.debug("Re-try.")
                        test_pred_arr = self.forward_propagation(
                            test_batch_observed_arr
                        )

                    if self.__save_flag is True:
                        np.save("test_pred_arr_" + str(epoch), test_pred_arr)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Convolutional Auto-Encoder's loss:")
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=ver_pred_arr[:, -1], 
                                train_label_arr=batch_target_arr[:, -1],
                                test_pred_arr=test_pred_arr[:, -1],
                                test_label_arr=test_batch_target_arr[:, -1]
                            )
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Encoder/Decoder's loss: ")
                            self.__logger.debug("Training: " + str(self.__encoder_decoder_loss) + " Test: " + str(self.__test_encoder_decoder_loss))
                            self.__logger.debug("-" * 100)

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Eary stopping.")
            eary_stop_flag = False

        self.__remember_best_params(best_weight_params_list, best_bias_params_list)
        self.__learn_flag = False
        self.__logger.debug("end. ")

    def __remember_best_params(self, best_weight_params_list, best_bias_params_list):
        '''
        Remember best parameters.
        
        Args:
            best_weight_params_list:    `list` of weight parameters.
            best_bias_params_list:      `list` of bias parameters.

        '''
        if len(best_weight_params_list) and len(best_bias_params_list):
            for i in range(len(self.__layerable_cnn_list)):
                self.__layerable_cnn_list[i].graph.weight_arr = best_weight_params_list[i]
                self.__layerable_cnn_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Convolutional Auto-Encoder's best params are saved.")
        else:
            self.__logger.debug("Convolutional Auto-Encoder's best params are not saved.")
            
        if len(self.__encoder_best_params_list) and len(self.__decoder_best_params_list):
            self.__encoder.graph.weights_output_arr = self.__encoder_best_params_list[0]
            self.__encoder.graph.output_bias_arr = self.__encoder_best_params_list[1]
            self.__encoder.graph.weights_lstm_hidden_arr = self.__encoder_best_params_list[2]
            self.__encoder.graph.weights_lstm_observed_arr = self.__encoder_best_params_list[3]
            self.__encoder.graph.lstm_bias_arr = self.__encoder_best_params_list[4]

            self.__decoder.graph.weights_output_arr = self.__decoder_best_params_list[0]
            self.__decoder.graph.output_bias_arr = self.__decoder_best_params_list[1]
            self.__decoder.graph.weights_lstm_hidden_arr = self.__decoder_best_params_list[2]
            self.__decoder.graph.weights_lstm_observed_arr = self.__decoder_best_params_list[3]
            self.__decoder.graph.lstm_bias_arr = self.__decoder_best_params_list[4]

            self.__logger.debug("Encoder/Decoder's best params are saved.")
        else:
            self.__logger.debug("Encoder/Decoder's best params are not saved.")

    def inference(self, np.ndarray[DOUBLE_t, ndim=5] observed_arr):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data points.

        Returns:
            Predicted array like or sparse matrix.
        '''
        cdef np.ndarray[DOUBLE_t, ndim=5] pred_arr = self.forward_propagation(
            observed_arr
        )
        return pred_arr

    def temporal_inference(
        self,
        np.ndarray observed_arr,
        np.ndarray hidden_activity_arr=None,
        np.ndarray rnn_activity_arr=None
    ):
        r'''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data ponts.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            rnn_activity_arr:       Array like or sparse matrix as the state in RNN.

        Returns:
            Tuple(
                Array like or sparse matrix of reconstructed instances of time-series,
                Array like or sparse matrix of the state in hidden layer,
                Array like or sparse matrix of the state in RNN
            )
        '''
        if hidden_activity_arr is not None:
            self.__encoder.graph.hidden_activity_arr = hidden_activity_arr
        else:
            self.__encoder.graph.hidden_activity_arr = np.array([])

        if rnn_activity_arr is not None:
            self.__encoder.graph.rnn_activity_arr = rnn_activity_arr
        else:
            self.__encoder.graph.rnn_activity_arr = np.array([])

        _ = self.__encoder.inference(observed_arr)
        encoded_arr = self.__encoder.get_feature_points()[:, ::-1, :]
        _ = self.__decoder.inference(
            encoded_arr,
        )
        decoded_arr = self.__decoder.get_feature_points()[:, ::-1, :]
        
        self.__encoded_features_arr = encoded_arr
        self.__temporal_reconstruction_error_arr = self.__computable_loss.compute_loss(
            decoded_arr[:, -1],
            observed_arr[:, -1]
        )
        return decoded_arr

    def forward_propagation(self, np.ndarray[DOUBLE_t, ndim=5] img_arr):
        '''
        Forward propagation in Convolutional Auto-Encoder.
        
        Override.
        
        Args:
            img_arr:    `np.ndarray` of image file array.
        
        Returns:
            Propagated `np.ndarray`.
        '''
        cdef int i = 0
        cdef np.ndarray conv_arr = None
        cdef np.ndarray[DOUBLE_t, ndim=4] conv_output_arr = None
        for seq in range(img_arr.shape[1]):
            for i in range(len(self.__layerable_cnn_list)):
                try:
                    if i == 0:
                        conv_output_arr = self.__layerable_cnn_list[i].convolve(img_arr[:, seq])
                    else:
                        conv_output_arr = self.__layerable_cnn_list[i].convolve(conv_output_arr)
                except:
                    self.__logger.debug("Error raised in Convolution layer " + str(i + 1))
                    raise

            if conv_arr is None:
                conv_arr = np.expand_dims(conv_output_arr, axis=0)
            else:
                conv_arr = np.r_[conv_arr, np.expand_dims(conv_output_arr, axis=0)]

        conv_arr = conv_arr.transpose((2, 0, 1, 3, 4))

        cdef int sample_n = conv_arr.shape[0]
        cdef int seq_len = conv_arr.shape[1]
        cdef int channel = conv_arr.shape[2]
        cdef int width = conv_arr.shape[3]
        cdef int height = conv_arr.shape[4]

        cdef np.ndarray decoded_arr
        cdef np.ndarray delta_arr
        cdef np.ndarray encoder_delta_arr
        cdef np.ndarray decoder_delta_arr

        cdef np.ndarray[DOUBLE_t, ndim=3] conv_input_arr = conv_arr.reshape((sample_n, seq_len, -1))
        cdef np.ndarray[DOUBLE_t, ndim=3] observed_arr = np.empty((sample_n, seq_len, self.__fully_connected_dim))

        if self.__fully_connected_activation is not None:
            if self.__fully_connected_weight_arr is None:
                self.__fully_connected_weight_arr = np.random.normal(
                    size=(
                        conv_arr.reshape((sample_n, seq_len, -1)).shape[-1], 
                        self.__fully_connected_dim
                    )
                ) * 0.1

            for seq in range(conv_input_arr.shape[1]):
                observed_arr[:, seq] = self.__fully_connected_activation.activate(
                    np.dot(conv_input_arr[:, seq], self.__fully_connected_weight_arr)
                )

            decoded_arr = self.temporal_inference(observed_arr)
            self.__decoded_features_arr = decoded_arr
            ver_decoded_arr = decoded_arr.copy()
            loss = self.__temporal_reconstruction_error_arr
            delta_arr = self.__computable_loss.compute_delta(decoded_arr[:, 0], observed_arr[:, 0])
        else:
            decoded_arr = self.temporal_inference(conv_arr)
            self.__decoded_features_arr = decoded_arr
            loss = self.__temporal_reconstruction_error_arr
            delta_arr = self.__computable_loss.compute_delta(decoded_arr[:, 0], conv_arr[:, 0])

        if self.__learn_flag is True:
            self.__encoder_decoder_loss = loss
        else:
            self.__test_encoder_decoder_loss = loss

        if self.__learn_flag is True:
            self.__logger.debug("Encoder/Decoder's deltas are propagated.")
            decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.temporal_back_propagation(delta_arr)
            self.temporal_optimize(decoder_grads_list, encoder_grads_list, self.__now_learning_rate, self.__now_epoch)

            if self.__temporal_min_loss is None or self.__temporal_min_loss > self.__encoder_decoder_loss:
                self.__temporal_min_loss = self.__encoder_decoder_loss
                self.__encoder_best_params_list = [
                    self.__encoder.graph.weights_lstm_hidden_arr,
                    self.__encoder.graph.weights_lstm_observed_arr,
                    self.__encoder.graph.lstm_bias_arr
                ]
                self.__decoder_best_params_list = [
                    self.__decoder.graph.weights_lstm_hidden_arr,
                    self.__decoder.graph.weights_lstm_observed_arr,
                    self.__decoder.graph.lstm_bias_arr
                ]

                self.__logger.debug("Encoder/Decoder's best params are updated.")

        self.__encoder.graph.hidden_activity_arr = np.array([])
        self.__encoder.graph.rnn_activity_arr = np.array([])
        self.__decoder.graph.hidden_activity_arr = np.array([])
        self.__decoder.graph.rnn_activity_arr = np.array([])

        conv_arr = (conv_arr - conv_arr.min()) / (conv_arr.max() - conv_arr.min())

        cdef np.ndarray[DOUBLE_t, ndim=3] lstm_input_arr
        if self.__fully_connected_activation is not None:
            lstm_input_arr = np.empty(
                (
                    sample_n, 
                    seq_len, 
                    conv_arr.reshape((sample_n, seq_len, -1)).shape[-1]
                )
            )
            for seq in range(decoded_arr.shape[1]):
                if decoded_arr[:, seq].shape[-1] != self.__fully_connected_weight_arr.T.shape[-1]:
                    lstm_input_arr[:, seq] = np.dot(decoded_arr[:, seq], self.__fully_connected_weight_arr.T)
                else:
                    lstm_input_arr[:, seq] = decoded_arr[:, seq]

            lstm_input_arr = (lstm_input_arr - lstm_input_arr.min()) / (lstm_input_arr.max() - lstm_input_arr.min())

            conv_arr = conv_arr + lstm_input_arr.reshape((sample_n, seq_len, channel, width, height))
            conv_arr = self.__fully_connected_activation.activate(conv_arr)
        else:
            if conv_arr.ndim == decoded_arr.ndim:
                conv_arr = np.tanh(conv_arr + decoded_arr)
            else:
                conv_arr = np.tanh(conv_arr + decoded_arr.reshape(conv_arr.copy().shape))

        conv_arr = conv_arr - conv_arr.mean()
        self.__spatio_temporal_features_arr = conv_arr

        layerable_cnn_list = self.__layerable_cnn_list[::-1]
        test_arr, _ = layerable_cnn_list[0].deconvolve(conv_arr[:, -1])

        cdef np.ndarray deconv_arr = None
        cdef np.ndarray[DOUBLE_t, ndim=4] deconv_output_arr
        for seq in range(conv_arr.shape[1]):
            for i in range(len(layerable_cnn_list)):
                try:
                    if i == 0:
                        deconv_output_arr, _ = layerable_cnn_list[i].deconvolve(conv_arr[:, seq])
                    else:
                        deconv_output_arr, _ = layerable_cnn_list[i].deconvolve(deconv_output_arr)
                except:
                    self.__logger.debug("Error raised in Deconvolution layer " + str(i + 1))
                    raise

            if deconv_arr is None:
                deconv_arr = np.expand_dims(deconv_output_arr, axis=0)
            else:
                deconv_arr = np.r_[deconv_arr, np.expand_dims(deconv_output_arr, axis=0)]

        deconv_arr = deconv_arr.transpose((1, 0, 2, 3, 4))
        return deconv_arr

    def back_propagation(self, np.ndarray[DOUBLE_t, ndim=4] delta_arr):
        '''
        Back propagation in CNN.
        
        Args:
            Delta.
        
        Returns.
            Delta.
        '''
        cdef int i = 0

        for i in range(len(self.layerable_cnn_list)):
            try:
                delta_arr = self.layerable_cnn_list[i].convolve(delta_arr, no_bias_flag=True)
            except:
                self.__logger.debug("Backward raised error in Convolution layer " + str(i + 1))
                raise

        layerable_cnn_list = self.__layerable_cnn_list[::-1]
        for i in range(len(layerable_cnn_list)):
            try:
                delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)
            except:
                self.__logger.debug(
                    "Delta computation raised an error in CNN layer " + str(len(layerable_cnn_list) - i)
                )
                raise

        return delta_arr

    def optimize(self, double learning_rate, int epoch):
        '''
        Back propagation.
        
        Args:
            learning_rate:  Learning rate.
            epoch:          Now epoch.
            
        '''
        params_list = []
        grads_list = []
        for i in range(len(self.__layerable_cnn_list)):
            params_list.append(self.__layerable_cnn_list[i].graph.weight_arr)
            grads_list.append(self.__layerable_cnn_list[i].delta_weight_arr)

        for i in range(len(self.__layerable_cnn_list)):
            params_list.append(self.__layerable_cnn_list[i].graph.bias_arr)
            grads_list.append(self.__layerable_cnn_list[i].delta_bias_arr)

        params_list = self.__opt_params.optimize(
            params_list,
            grads_list,
            learning_rate
        )
        
        params_dict = {}
        i = 0
        for i in range(len(self.__layerable_cnn_list)):
            self.__layerable_cnn_list[i].graph.weight_arr = params_list.pop(0)
            if ((epoch + 1) % self.__attenuate_epoch == 0):
                self.__layerable_cnn_list[i].graph.weight_arr = self.__opt_params.constrain_weight(
                    self.__layerable_cnn_list[i].graph.weight_arr
                )

        for i in range(len(self.__layerable_cnn_list)):
            self.__layerable_cnn_list[i].graph.bias_arr = params_list.pop(0)

        for i in range(len(self.__layerable_cnn_list)):
            self.__layerable_cnn_list[i].reset_delta()

    def temporal_back_propagation(
        self,
        np.ndarray delta_arr
    ):
        r'''
        Back propagation in temporal Encoder/Decoder.

        Args:
            pred_arr:            `np.ndarray` of predicted data points from decoder.
            delta_output_arr:    Delta.
        
        Returns:
            Tuple(
                decoder's `list` of gradations,
                encoder's `np.ndarray` of Delta, 
                encoder's `list` of gradations,
            )
        '''
        decoder_delta_arr, decoder_grads_list = self.__decoder.hidden_back_propagate(delta_arr)
        decoder_grads_list.insert(0, None)
        decoder_grads_list.insert(0, None)
        encoder_delta_arr, encoder_grads_list = self.__encoder.hidden_back_propagate(decoder_delta_arr[:, -1])
        encoder_grads_list.insert(0, None)
        encoder_grads_list.insert(0, None)
        return decoder_grads_list, encoder_delta_arr, encoder_grads_list

    def temporal_optimize(
        self,
        decoder_grads_list,
        encoder_grads_list,
        double learning_rate,
        int epoch
    ):
        '''
        Back propagation in temporal Encoder/Decoder.
        
        Args:
            decoder_grads_list:     decoder's `list` of graduations.
            encoder_grads_list:     encoder's `list` of graduations.
            learning_rate:          Learning rate.
            epoch:                  Now epoch.
        '''
        self.__decoder.optimize(decoder_grads_list, learning_rate, epoch)
        self.__encoder.optimize(encoder_grads_list, learning_rate, epoch)

    def save_pre_learned_params(self, dir_path):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_path:   Path of dir. If `None`, the file is saved in the current directory.
        '''
        if dir_path[-1] != "/":
            dir_path = dir_path + "/"

        for i in range(len(self.layerable_cnn_list)):
            self.layerable_cnn_list[i].graph.save_pre_learned_params(dir_path + "spatio_cnn_" + str(i) + ".npz")
        self.__encoder.graph.save_pre_learned_params(dir_path + "temporal_encoder.npz")
        self.__decoder.graph.save_pre_learned_params(dir_path + "temporal_decoder.npz")

    def extract_features_points(self):
        '''
        Extract features points.

        Returns:
            Tuple(
                Temporal encoded feature points,
                Temporal decoded feature points,
                Fully-connected Spatio encoded feature points and Temporal decoded feature points  
            )
        '''
        return (
            self.__encoded_features_arr,
            self.__decoded_features_arr,
            self.__spatio_temporal_features_arr
        )

    def get_layerable_cnn_list(self):
        ''' getter '''
        return self.__layerable_cnn_list

    def set_layerable_cnn_list(self, value):
        ''' setter '''
        for layerable_cnn in value:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError()

        self.__layerable_cnn_list = value

    layerable_cnn_list = property(get_layerable_cnn_list, set_layerable_cnn_list)

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