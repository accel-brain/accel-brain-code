# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.layerable_cnn import LayerableCNN
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.optimization.opt_params import OptParams
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface
from pydbm.activation.tanh_function import TanhFunction
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
        save_flag=False
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
            save_flag:                      If `True`, save `np.ndarray` of inferenced test data in training.

        '''
        for layerable_cnn in layerable_cnn_list:
            if isinstance(layerable_cnn, LayerableCNN) is False:
                raise TypeError("The type of value of `layerable_cnn` must be `LayerableCNN`.")
        self.__layerable_cnn_list = layerable_cnn_list

        if isinstance(encoder, ReconstructableModel) is False:
            raise TypeError()
        if isinstance(decoder, ReconstructableModel) is False:
            raise TypeError()

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
            self.__fully_connected_activation = TanhFunction()

        self.__tol = tol

        self.__memory_tuple_list = []
        
        self.__save_flag = save_flag

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
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

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
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr
                            )
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

        if len(best_weight_params_list) and len(best_bias_params_list):
            for i in range(len(self.__layerable_cnn_list)):
                self.__layerable_cnn_list[i].graph.weight_arr = best_weight_params_list[i]
                self.__layerable_cnn_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Convolutional Auto-Encoder's best params are saved.")

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
                                train_pred_arr=ver_pred_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_pred_arr,
                                test_label_arr=test_batch_target_arr
                            )
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Encoder/Decoder's loss: " + str(self.__encoder_decoder_loss))
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

        if len(best_weight_params_list) and len(best_bias_params_list):
            for i in range(len(self.__layerable_cnn_list)):
                self.__layerable_cnn_list[i].graph.weight_arr = best_weight_params_list[i]
                self.__layerable_cnn_list[i].graph.bias_arr = best_bias_params_list[i]
            self.__logger.debug("Convolutional Auto-Encoder's best params are saved.")

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

        self.__logger.debug("end. ")

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

        conv_arr = conv_arr.transpose((1, 0, 2, 3, 4))

        cdef int sample_n = conv_arr.shape[0]
        cdef int seq_len = conv_arr.shape[1]
        cdef int channel = conv_arr.shape[2]
        cdef int width = conv_arr.shape[3]
        cdef int height = conv_arr.shape[4]

        cdef np.ndarray[DOUBLE_t, ndim=3] hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] encoded_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] decoded_arr
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] encoder_delta_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] decoder_delta_arr

        cdef np.ndarray[DOUBLE_t, ndim=3] conv_input_arr = conv_arr.reshape((sample_n, seq_len, -1))
        cdef np.ndarray[DOUBLE_t, ndim=3] observed_arr = np.empty((sample_n, seq_len, self.__fully_connected_dim))

        if self.__fully_connected_weight_arr is None:
            self.__fully_connected_weight_arr = np.random.normal(
                size=(
                    conv_arr.reshape((sample_n, seq_len, -1)).shape[-1], 
                    self.__fully_connected_dim
                )
            ) * 0.1

        for seq in range(conv_arr.shape[1]):
            observed_arr[:, seq] = self.__fully_connected_activation.activate(
                np.dot(conv_input_arr[:, seq], self.__fully_connected_weight_arr)
            )

        encoded_arr = self.__encoder.inference(observed_arr)
        decoded_arr = self.__decoder.inference(
            self.__encoder.get_feature_points()[:, ::-1, :]
        )
        hidden_activity_arr = self.__decoder.get_feature_points()[:, ::-1, :]
        ver_hidden_activity_arr = hidden_activity_arr.copy()
        delta_arr = self.__computable_loss.compute_delta(
            hidden_activity_arr[:, 0, :],
            observed_arr[:, 0, :]
        )
        loss = self.__computable_loss.compute_loss(
            hidden_activity_arr[:, 0, :],
            observed_arr[:, 0, :]
        )

        self.__encoder_decoder_loss = loss

        decoder_delta_arr, decoder_lstm_grads_list = self.__decoder.hidden_back_propagate(
            delta_arr
        )
        encoder_delta_arr, encoder_lstm_grads_list = self.__encoder.hidden_back_propagate(
            decoder_delta_arr[:, 0, :]
        )
        decoder_grads_list = [None, None]
        [decoder_grads_list.append(d) for d in decoder_lstm_grads_list]
        encoder_grads_list = [None, None]
        [encoder_grads_list.append(d) for d in encoder_lstm_grads_list]
        self.__decoder.optimize(decoder_grads_list, self.__now_learning_rate, self.__now_epoch)
        self.__encoder.optimize(encoder_grads_list, self.__now_learning_rate, self.__now_epoch)

        if self.__temporal_min_loss is None or self.__temporal_min_loss > loss:
            self.__temporal_min_loss = loss
            self.__encoder_best_params_list = [
                self.__encoder.graph.weights_output_arr,
                self.__encoder.graph.output_bias_arr,
                self.__encoder.graph.weights_lstm_hidden_arr,
                self.__encoder.graph.weights_lstm_observed_arr,
                self.__encoder.graph.lstm_bias_arr
            ]
            self.__decoder_best_params_list = [
                self.__decoder.graph.weights_output_arr,
                self.__decoder.graph.output_bias_arr,
                self.__decoder.graph.weights_lstm_hidden_arr,
                self.__decoder.graph.weights_lstm_observed_arr,
                self.__decoder.graph.lstm_bias_arr
            ]

            self.__logger.debug("Encoder/Decoder's best params are updated.")

        self.__encoder.graph.hidden_activity_arr = np.array([])
        self.__encoder.graph.rnn_activity_arr = np.array([])
        self.__decoder.graph.hidden_activity_arr = np.array([])
        self.__decoder.graph.rnn_activity_arr = np.array([])
        
        cdef np.ndarray[DOUBLE_t, ndim=3] lstm_input_arr = np.empty((
            sample_n, 
            seq_len, 
            conv_arr.reshape((sample_n, seq_len, -1)).shape[-1])
        )

        for seq in range(hidden_activity_arr.shape[1]):
            lstm_input_arr[:, seq] = np.dot(hidden_activity_arr[:, seq], self.__fully_connected_weight_arr.T)

        conv_arr = (conv_arr - conv_arr.min()) / (conv_arr.max() - conv_arr.min())
        lstm_input_arr = (lstm_input_arr - lstm_input_arr.min()) / (lstm_input_arr.max() - lstm_input_arr.min())

        conv_arr = conv_arr + lstm_input_arr.reshape((sample_n, seq_len, channel, width, height))
        conv_arr = conv_arr - conv_arr.mean()
        conv_arr = self.__fully_connected_activation.activate(conv_arr)

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
        self.__logger.debug("-" * 100)
        for i in range(len(layerable_cnn_list)):
            try:
                self.__logger.debug("Input delta shape in CNN layer: " + str(len(layerable_cnn_list) - i))
                self.__logger.debug((
                    delta_arr.shape[0],
                    delta_arr.shape[1],
                    delta_arr.shape[2],
                    delta_arr.shape[3]
                ))

                delta_arr = layerable_cnn_list[i].back_propagate(delta_arr)

            except:
                self.__logger.debug(
                    "Delta computation raised an error in CNN layer " + str(len(layerable_cnn_list) - i)
                )
                raise

        self.__logger.debug("-" * 100)
        self.__logger.debug("Propagated delta shape in CNN layer: " + str(len(layerable_cnn_list) - i))
        self.__logger.debug((
            delta_arr.shape[0],
            delta_arr.shape[1],
            delta_arr.shape[2],
            delta_arr.shape[3]
        ))
        self.__logger.debug("-" * 100)
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
