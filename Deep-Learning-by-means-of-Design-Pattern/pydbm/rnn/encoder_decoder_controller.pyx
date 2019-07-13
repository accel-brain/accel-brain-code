# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.rnn.interface.reconstructable_model import ReconstructableModel
from pydbm.cnn.feature_generator import FeatureGenerator
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.verification.interface.verificatable_result import VerificatableResult
ctypedef np.float64_t DOUBLE_t


class EncoderDecoderController(object):
    '''
    Encoder/Decoder based on LSTM networks.
    
    This library provides Encoder/Decoder based on LSTM, 
    which is a reconstruction model and makes it possible to extract 
    series features embedded in deeper layers. The LSTM encoder learns 
    a fixed length vector of time-series observed data points and the 
    LSTM decoder uses this representation to reconstruct the time-series 
    using the current hidden state and the value inferenced at the previous time-step.
    
    One interesting application example is the Encoder/Decoder for 
    Anomaly Detection (EncDec-AD) paradigm (Malhotra, P., et al. 2016).
    This reconstruction model learns to reconstruct normal time-series behavior, 
    and thereafter uses reconstruction error to detect anomalies. 
    Malhotra, P., et al. (2016) showed that EncDec-AD paradigm is robust 
    and can detect anomalies from predictable, unpredictable, periodic, aperiodic, 
    and quasi-periodic time-series. Further, they showed that the paradigm is able 
    to detect anomalies from short time-series (length as small as 30) as well as 
    long time-series (length as large as 500).

    References:
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_sine_wave_prediction_by_LSTM_encoder_decoder.ipynb
        - https://github.com/chimera0/accel-brain-code/blob/master/Deep-Learning-by-means-of-Design-Pattern/demo/demo_anomaly_detection_by_enc_dec_ad.ipynb
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
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
        dropout_rate=0.5,
        tol=1e-04,
        tld=100.0
    ):
        '''
        Init.
        
        Args:
            encoder:                        is-a `ReconstructableModel`.
            decoder:                        is-a `ReconstructableModel`.
            computable_loss:                is-a `ComputableLoss`.
            verificatable_result:           is-a `VerificatableResult`.
            epochs:                         Epochs of mini-batch.
            bath_size:                      Batch size of mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.
            dropout_rate:                   The propability of dropout.
            tol:                            Tolerance for the optimization.
                                            When the loss or score is not improving by at least tol 
                                            for two consecutive iterations, convergence is considered 
                                            to be reached and training stops.

            tld:                            Tolerance for deviation of loss.

        '''
        if isinstance(encoder, ReconstructableModel) is False:
            raise TypeError()
        if isinstance(decoder, ReconstructableModel) is False:
            raise TypeError()
        if isinstance(computable_loss, ComputableLoss) is False:
            raise TypeError()
        if isinstance(verificatable_result, VerificatableResult) is False:
            raise TypeError()

        self.__encoder = encoder
        self.__decoder = decoder
        self.__computable_loss = computable_loss
        self.__epochs = epochs
        self.__batch_size = batch_size

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__test_size_rate = test_size_rate

        self.__dropout_rate = dropout_rate
        self.__verificatable_result = verificatable_result

        self.__tol = tol
        self.__tld = tld

        logger = getLogger("pydbm")
        self.__logger = logger

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points
        for vector representation of the input time-series.

        Override.

        Args:
            observed_arr:       Array like or sparse matrix as the observed data ponts.
            target_arr:         Array like or sparse matrix as the target data points.
                                To learn as Auto-encoder, this value must be `None` or equivalent to `observed_arr`.
        '''
        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray batch_observed_arr
        cdef np.ndarray batch_target_arr

        if row_t != 0 and row_t != row_o:
            raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if row_t == 0:
            target_arr = observed_arr.copy()
        else:
            if target_arr.ndim == 2:
                target_arr = target_arr.reshape((target_arr.shape[0], 1, target_arr.shape[1]))

        if self.__test_size_rate > 0:
            train_index = np.random.choice(observed_arr.shape[0], round((1 - self.__test_size_rate) * observed_arr.shape[0]), replace=False)
            test_index = np.array(list(set(range(observed_arr.shape[0])) - set(train_index)))
            train_observed_arr = observed_arr[train_index]
            test_observed_arr = observed_arr[test_index]
            train_target_arr = target_arr[train_index]
            test_target_arr = target_arr[test_index]
        else:
            train_observed_arr = observed_arr
            train_target_arr = target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray decoded_arr
        cdef np.ndarray delta_arr
        cdef np.ndarray encoder_delta_arr

        encoder_best_params_list = []
        decoder_best_params_list = []
        try:
            self.__change_inferencing_mode(False)
            self.__memory_tuple_list = []
            eary_stop_flag = False
            loss_list = []
            min_loss = None
            for epoch in range(self.__epochs):
                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = train_observed_arr[rand_index]
                batch_target_arr = train_target_arr[rand_index]
                try:
                    decoded_arr = self.inference(batch_observed_arr)
                    loss = self.__reconstruction_error_arr.mean()
                    train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    loss = loss + train_weight_decay
                    
                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(encoder_best_params_list, decoder_best_params_list)
                        # Re-try.
                        decoded_arr = self.inference(batch_observed_arr)
                        loss = self.__reconstruction_error_arr.mean()

                        train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                        loss = loss + train_weight_decay

                    delta_arr = self.__computable_loss.compute_delta(
                        decoded_arr, 
                        batch_target_arr
                    )

                    decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.back_propagation(delta_arr)
                    self.optimize(decoder_grads_list, encoder_grads_list, learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        encoder_best_params_list = [
                            self.__encoder.graph.weights_lstm_hidden_arr,
                            self.__encoder.graph.weights_lstm_observed_arr,
                            self.__encoder.graph.lstm_bias_arr
                        ]
                        decoder_best_params_list = [
                            self.__decoder.graph.weights_lstm_hidden_arr,
                            self.__decoder.graph.weights_lstm_observed_arr,
                            self.__decoder.graph.lstm_bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                    self.__encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder.graph.cec_activity_arr = np.array([])
                    self.__decoder.graph.hidden_activity_arr = np.array([])
                    self.__decoder.graph.cec_activity_arr = np.array([])

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated."
                        )
                        raise

                if self.__test_size_rate > 0:
                    rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                    test_batch_observed_arr = test_observed_arr[rand_index]
                    test_batch_target_arr = test_target_arr[rand_index]

                    self.__change_inferencing_mode(True)
                    test_decoded_arr = self.inference(test_batch_observed_arr)
                    test_loss = self.__reconstruction_error_arr.mean()
                    test_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    test_loss = test_loss + test_weight_decay

                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(test_loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(encoder_best_params_list, decoder_best_params_list)
                        # Re-try.
                        test_decoded_arr = self.inference(test_batch_observed_arr)
                        test_loss = self.__reconstruction_error_arr.mean()

                    test_loss += self.__encoder.weight_decay_term + self.__decoder.weight_decay_term

                    self.__change_inferencing_mode(False)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=decoded_arr, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_decoded_arr,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_weight_decay,
                                test_penalty=test_weight_decay
                            )
                    self.__encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder.graph.cec_activity_arr = np.array([])
                    self.__decoder.graph.hidden_activity_arr = np.array([])
                    self.__decoder.graph.cec_activity_arr = np.array([])

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break
                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Early stopping.")
            eary_stop_flag = False
        
        self.__remember_best_params(encoder_best_params_list, decoder_best_params_list)
        self.__change_inferencing_mode(True)
        self.__logger.debug("end. ")

    def learn_generated(self, feature_generator):
        '''
        Learn features generated by `FeatureGenerator`.
        
        Args:
            feature_generator:    is-a `FeatureGenerator`.

        '''
        if isinstance(feature_generator, FeatureGenerator) is False:
            raise TypeError("The type of `feature_generator` must be `FeatureGenerator`.")

        cdef double learning_rate = self.__learning_rate
        cdef int epoch
        cdef int batch_index

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray test_observed_arr
        cdef np.ndarray test_target_arr

        cdef np.ndarray rand_index
        cdef np.ndarray batch_observed_arr
        cdef np.ndarray batch_target_arr

        cdef double loss
        cdef double test_loss
        cdef np.ndarray hidden_activity_arr
        cdef np.ndarray decoded_arr
        cdef np.ndarray test_decoded_arr
        cdef np.ndarray delta_arr
        cdef np.ndarray encoder_delta_arr

        encoder_best_params_list = []
        decoder_best_params_list = []
        try:
            self.__change_inferencing_mode(False)
            self.__memory_tuple_list = []
            eary_stop_flag = False
            loss_list = []
            min_loss = None
            epoch = 0
            for batch_observed_arr, batch_target_arr, test_batch_observed_arr, test_batch_target_arr in feature_generator.generate():
                epoch += 1

                if ((epoch + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate

                try:
                    decoded_arr = self.inference(batch_observed_arr)
                    loss = self.__reconstruction_error_arr.mean()
                    train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    loss = loss + train_weight_decay
                    
                    remember_flag = False
                    if len(loss_list) > 0:
                        if abs(loss - (sum(loss_list)/len(loss_list))) > self.__tld:
                            remember_flag = True

                    if remember_flag is True:
                        self.__remember_best_params(encoder_best_params_list, decoder_best_params_list)
                        # Re-try.
                        decoded_arr = self.inference(batch_observed_arr)
                        loss = self.__reconstruction_error_arr.mean()
                        train_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                        loss = loss + train_weight_decay

                    delta_arr = self.__computable_loss.compute_delta(
                        decoded_arr, 
                        batch_target_arr
                    )

                    decoder_grads_list, encoder_delta_arr, encoder_grads_list = self.back_propagation(delta_arr)
                    self.optimize(decoder_grads_list, encoder_grads_list, learning_rate, epoch)

                    if min_loss is None or min_loss > loss:
                        min_loss = loss
                        encoder_best_params_list = [
                            self.__encoder.graph.weights_lstm_hidden_arr,
                            self.__encoder.graph.weights_lstm_observed_arr,
                            self.__encoder.graph.lstm_bias_arr
                        ]
                        decoder_best_params_list = [
                            self.__decoder.graph.weights_lstm_hidden_arr,
                            self.__decoder.graph.weights_lstm_observed_arr,
                            self.__decoder.graph.lstm_bias_arr
                        ]
                        self.__logger.debug("Best params are updated.")

                    self.__encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder.graph.cec_activity_arr = np.array([])
                    self.__decoder.graph.hidden_activity_arr = np.array([])
                    self.__decoder.graph.cec_activity_arr = np.array([])

                except FloatingPointError:
                    if epoch > int(self.__epochs * 0.7):
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated. Because of early stopping, this error is catched and the parameter is not updated."
                        )
                        eary_stop_flag = True
                        break
                    else:
                        self.__logger.debug(
                            "Underflow occurred when the parameters are being updated."
                        )
                        raise

                if self.__test_size_rate > 0:
                    self.__change_inferencing_mode(True)
                    test_decoded_arr = self.inference(test_batch_observed_arr)
                    test_loss = self.__reconstruction_error_arr.mean()
                    test_weight_decay = self.__encoder.weight_decay_term + self.__decoder.weight_decay_term
                    test_loss = test_loss + test_weight_decay

                    self.__change_inferencing_mode(False)

                    if self.__verificatable_result is not None:
                        if self.__test_size_rate > 0:
                            self.__verificatable_result.verificate(
                                self.__computable_loss,
                                train_pred_arr=decoded_arr + train_weight_decay, 
                                train_label_arr=batch_target_arr,
                                test_pred_arr=test_decoded_arr + test_weight_decay,
                                test_label_arr=test_batch_target_arr,
                                train_penalty=train_weight_decay,
                                test_penalty=test_weight_decay
                            )

                    self.__encoder.graph.hidden_activity_arr = np.array([])
                    self.__encoder.graph.cec_activity_arr = np.array([])
                    self.__decoder.graph.hidden_activity_arr = np.array([])
                    self.__decoder.graph.cec_activity_arr = np.array([])

                if epoch > 1 and abs(loss - loss_list[-1]) < self.__tol:
                    eary_stop_flag = True
                    break

                loss_list.append(loss)

        except KeyboardInterrupt:
            self.__logger.debug("Interrupt.")

        if eary_stop_flag is True:
            self.__logger.debug("Early stopping.")
            eary_stop_flag = False
        
        self.__remember_best_params(encoder_best_params_list, decoder_best_params_list)
        self.__change_inferencing_mode(True)

        self.__logger.debug("end. ")

    def __remember_best_params(self, encoder_best_params_list, decoder_best_params_list):
        '''
        Remember best parameters.
        
        Args:
            encoder_best_params_list:    `list` of encoder's parameters.
            decoder_best_params_list:    `list` of decoder's parameters.

        '''
        if len(encoder_best_params_list) > 0 and len(decoder_best_params_list) > 0:
            self.__encoder.graph.weights_lstm_hidden_arr = encoder_best_params_list[0]
            self.__encoder.graph.weights_lstm_observed_arr = encoder_best_params_list[1]
            self.__encoder.graph.lstm_bias_arr = encoder_best_params_list[2]

            self.__decoder.graph.weights_lstm_hidden_arr = decoder_best_params_list[0]
            self.__decoder.graph.weights_lstm_observed_arr = decoder_best_params_list[1]
            self.__decoder.graph.lstm_bias_arr = decoder_best_params_list[2]

            self.__logger.debug("Best params are saved.")

    def inference(
        self,
        np.ndarray observed_arr,
        np.ndarray hidden_activity_arr=None,
        np.ndarray cec_activity_arr=None
    ):
        '''
        Inference the feature points to reconstruct the time-series.

        Override.

        Args:
            observed_arr:           Array like or sparse matrix as the observed data ponts.
            hidden_activity_arr:    Array like or sparse matrix as the state in hidden layer.
            cec_activity_arr:       Array like or sparse matrix as the state in RNN.

        Returns:
            Tuple data.
            - Array like or sparse matrix of reconstructed instances of time-series,
            - Array like or sparse matrix of the state in hidden layer,
            - Array like or sparse matrix of the state in RNN.
        '''
        if hidden_activity_arr is not None:
            self.__encoder.graph.hidden_activity_arr = hidden_activity_arr
        else:
            self.__encoder.graph.hidden_activity_arr = np.array([])

        if cec_activity_arr is not None:
            self.__encoder.graph.cec_activity_arr = cec_activity_arr
        else:
            self.__encoder.graph.cec_activity_arr = np.array([])

        _ = self.__encoder.inference(observed_arr)
        encoded_arr = self.__encoder.get_feature_points()
        self.__feature_points_arr = encoded_arr
        decoded_arr = self.__decoder.inference(np.expand_dims(encoded_arr[:, -1], axis=1))
        decoded_arr = decoded_arr[:, ::-1]
        self.__reconstruction_error_arr = self.__computable_loss.compute_delta(
            decoded_arr,
            observed_arr
        )
        self.__pred_arr = decoded_arr
        return decoded_arr

    def back_propagation(self, np.ndarray delta_arr):
        '''
        Back propagation.

        Args:
            delta_output_arr:    Delta.
        
        Returns:
            Tuple data.
            - decoder's `list` of gradations,
            - encoder's `np.ndarray` of Delta, 
            - encoder's `list` of gradations.
        '''
        decoder_delta_arr, decoder_grads_list = self.__decoder.back_propagation(
            pred_arr=self.__pred_arr,
            delta_arr=delta_arr
        )
        encoder_delta_arr, delta_hidden_arr, encoder_grads_list = self.__encoder.hidden_back_propagate(decoder_delta_arr[:, 0])
        encoder_grads_list.insert(0, None)
        encoder_grads_list.insert(0, None)

        return decoder_grads_list, encoder_delta_arr, encoder_grads_list

    def optimize(
        self,
        decoder_grads_list,
        encoder_grads_list,
        double learning_rate,
        int epoch
    ):
        '''
        Back propagation.
        
        Args:
            decoder_grads_list:     decoder's `list` of graduations.
            encoder_grads_list:     encoder's `list` of graduations.
            learning_rate:          Learning rate.
            epoch:                  Now epoch.
        '''
        self.__decoder.optimize(decoder_grads_list, learning_rate, epoch)
        self.__encoder.optimize(encoder_grads_list, learning_rate, epoch)

    def __change_inferencing_mode(self, inferencing_mode):
        '''
        Change dropout rate in Encoder/Decoder.
        
        Args:
            dropout_rate:   The probalibity of dropout.
        '''
        self.__decoder.opt_params.inferencing_mode = inferencing_mode
        self.__encoder.opt_params.inferencing_mode = inferencing_mode

    def get_feature_points(self):
        '''
        Extract the activities in hidden layer and reset it, 
        considering this method will be called per one cycle in instances of time-series.

        Returns:
            The array like or sparse matrix of feature points.
        '''
        return self.__feature_points_arr

    def get_reconstruction_error(self):
        '''
        Extract the reconstructed error in inferencing.
        
        Returns:
            The array like or sparse matrix of reconstruction error. 
        '''
        return self.__reconstruction_error_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_encoder(self):
        ''' getter '''
        return self.__encoder

    encoder = property(get_encoder, set_readonly)

    def get_decoder(self):
        ''' getter '''
        return self.__decoder

    decoder = property(get_decoder, set_readonly)

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

    def save_pre_learned_params(self, dir_path):
        '''
        Save pre-learned parameters.
        
        Args:
            dir_path:   Path of dir. If `None`, the file is saved in the current directory.
        '''
        if dir_path[-1] != "/":
            dir_path = dir_path + "/"

        self.__encoder.save_pre_learned_params(dir_path, "encoder")
        self.__decoder.save_pre_learned_params(dir_path, "decoder")

    def load_pre_learned_params(self, dir_path):
        '''
        Load pre-learned parameters.

        If you want to load pre-learned parameters simultaneously with stacked graphs,
        call method `stack_graph` and setup the graphs before calling this method.
        
        Args:
            dir_path:    Dir path.
        '''
        if dir_path[-1] != "/":
            dir_path = dir_path + "/"

        self.__encoder.load_pre_learned_params(dir_path, "encoder")
        self.__decoder.load_pre_learned_params(dir_path, "decoder")

    def get_computable_loss(self):
        ''' getter '''
        return self.__computable_loss
    
    def set_computable_loss(self, value):
        ''' setter '''
        self.__computable_loss = value
    
    computable_loss = property(get_computable_loss, set_computable_loss)
