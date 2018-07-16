# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
from pydbm.rnn.interface.reconstructable_feature import ReconstructableFeature
ctypedef np.float64_t DOUBLE_t


class EncoderDecoderController(object):
    '''
    Encoder-Decoder scheme.
    '''
    
    # Encoder.
    __encoder = None

    # Decoder.
    __decoder = None

    # The list of reconsturcted error.
    __reconstructed_error_list = []

    def __init__(
        self,
        encoder,
        decoder,
        epochs,
        batch_size,
        learning_rate,
        learning_attenuate_rate,
        attenuate_epoch,
        test_size_rate=0.3,
        verificatable_result=None
    ):
        '''
        Init.

        Args:
            encoder:                    is-a `ReconstructableFeature` for vector representation of the input time-series.
            decoder:                    is-a `ReconstructableFeature` to reconstruct the time-series.

            epochs:                     Epochs of Mini-batch.
            bath_size:                  Batch size of Mini-batch.
            learning_rate:              Learning rate.
            learning_attenuate_rate:    Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:            Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
            test_size_rate:             Size of Test data set. If this value is `0`, the validation will not be executed.

        '''
        if isinstance(encoder, ReconstructableFeature):
            self.encoder = encoder
        else:
            raise TypeError()

        if isinstance(decoder, ReconstructableFeature):
            self.decoder = decoder
        else:
            raise TypeError()

        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

        self.__test_size_rate = test_size_rate
        if isinstance(verificatable_result, VerificatableResult):
            self.__verificatable_result = verificatable_result
        else:
            raise TypeError()

        logger = getLogger("pydbm")
        self.__logger = logger
        self.__logger.debug("encoder_decoder_controller is started. ")

    def learn(self, np.ndarray[DOUBLE_t, ndim=3] observed_arr, np.ndarray target_arr=np.array([])):
        '''
        Learn the observed data points.
        for vector representation of the input time-series.

        In Encoder-Decode scheme, usecase of this method may be pre-training with `__learned_params_dict`.

        Override.

        Args:
            observed_arr:    Array like or sparse matrix as the observed data ponts.
            target_arr:      Array like or sparse matrix as the target data points.

        '''
        self.__logger.debug("encoder_decoder_controler.learn is started. ")

        cdef int row_o = observed_arr.shape[0]
        cdef int row_t = target_arr.shape[0]

        cdef np.ndarray train_index
        cdef np.ndarray test_index
        cdef np.ndarray[DOUBLE_t, ndim=3] train_observed_arr
        cdef np.ndarray train_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=3] test_observed_arr
        cdef np.ndarray test_target_arr

        cdef double learning_rate = self.__learning_rate

        cdef int epoch
        cdef np.ndarray rand_index
        cdef np.ndarray[DOUBLE_t, ndim=3] batch_observed_arr
        cdef np.ndarray batch_target_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] rnn_activity_arr

        cdef np.ndarray[DOUBLE_t, ndim=1] _output_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] _hidden_activity_arr
        cdef np.ndarray[DOUBLE_t, ndim=1] _rnn_activity_arr

        cdef np.ndarray input_arr
        cdef np.ndarray rnn_arr
        cdef np.ndarray output_arr
        cdef np.ndarray label_arr
        cdef int batch_index
        cdef np.ndarray[DOUBLE_t, ndim=2] time_series_X

        cdef np.ndarray test_output_arr
        cdef np.ndarray test_label_arr

        if row_t == 0:
            target_arr = observed_arr.copy()
        else:
            if target_arr.ndim == 2:
                target_arr = target_arr.reshape((target_arr.shape[0], 1, target_arr.shape[1]))

            if row_o != row_t:
                raise ValueError("The row of `target_arr` must be equivalent to the row of `observed_arr`.")

        if target_arr is None:
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

        self.__reconstructed_error_list = []
        self.__labeled_list = []

        for epoch in range(self.__epochs):
            if ((epoch + 1) % self.__attenuate_epoch == 0):
                learning_rate = learning_rate / self.__learning_attenuate_rate

            rand_index = np.random.choice(train_observed_arr.shape[0], size=self.__batch_size)
            batch_observed_arr = train_observed_arr[rand_index]
            batch_target_arr = train_target_arr[rand_index]

            input_arr = None
            rnn_arr = None
            hidden_arr = None
            output_arr = None
            label_arr = None
            test_input_arr = None
            reconstructed_arr = None
            test_reconstructed_arr = None
            train_labeled = None
            test_labeled = None

            for batch_index in range(batch_observed_arr.shape[0]):
                time_series_X = batch_observed_arr[batch_index]
                target_time_series_X = batch_target_arr[batch_index]

                if input_arr is None:
                    input_arr = time_series_X[-1]
                else:
                    input_arr = np.vstack([input_arr, time_series_X[-1]])
                if label_arr is None:
                    label_arr = target_time_series_X[-1]
                else:
                    label_arr = np.vstack([label_arr, target_time_series_X[-1]])

                encoder_outputs = self.encoder.inference(time_series_X)
                feature_points_arr = self.encoder.get_feature_points()
                feature_points_arr = feature_points_arr.T
                feature_points_arr = feature_points_arr[::-1]
                feature_points_arr = feature_points_arr.reshape((1, feature_points_arr.shape[0]))

                decoder_outputs = self.decoder.inference(feature_points_arr)
                reconstructed_arr = self.decoder.get_feature_points().T

                for _ in range(time_series_X.shape[0]):
                    decoder_outputs = decoder_outputs.reshape((1, decoder_outputs.shape[0]))
                    decoder_outputs = self.decoder.inference(decoder_outputs)
                    reconstructed_arr = np.r_[reconstructed_arr, self.decoder.get_feature_points().T]

                if output_arr is None:
                    output_arr = reconstructed_arr
                else:
                    output_arr = np.vstack([output_arr, reconstructed_arr])

                self.encoder.back_propagation(target_time_series_X[-1])
                self.decoder.back_propagation(target_time_series_X[-1])

            self.encoder.update(learning_rate)
            self.decoder.update(learning_rate)

            if self.__test_size_rate > 0:
                rand_index = np.random.choice(test_observed_arr.shape[0], size=self.__batch_size)
                batch_observed_arr = test_observed_arr[rand_index]
                batch_target_arr = test_target_arr[rand_index]

                test_output_arr = None
                test_label_arr = None
                for batch_index in range(batch_observed_arr.shape[0]):
                    time_series_X_arr = batch_observed_arr[batch_index]
                    target_time_series_X_arr = batch_target_arr[batch_index]
                    if test_input_arr is None:
                        test_input_arr = time_series_X_arr
                    else:
                        test_input_arr = np.r_[test_input_arr, time_series_X_arr]

                    if test_label_arr is None:
                        test_label_arr = target_time_series_X_arr[-1]
                    else:
                        test_label_arr = np.vstack([test_label_arr, target_time_series_X_arr[-1]])

                    _test_output_arr = self.inference(time_series_X_arr)
                    if test_output_arr is None:
                        test_output_arr = _test_output_arr
                    else:
                        test_output_arr = np.vstack([test_output_arr, _test_output_arr])

                if self.__verificatable_result is not None:
                    if self.__test_size_rate > 0:
                        self.__verificatable_result.verificate(
                            train_pred_arr=output_arr,
                            train_label_arr=label_arr,
                            test_pred_arr=test_output_arr,
                            test_label_arr=test_label_arr
                        )


    def inference(self, np.ndarray[DOUBLE_t, ndim=2] time_series_X_arr):
        '''
        Inference the feature points
        to reconstruct the time-series.

        Args:
            time_series_X_arr:    Array like or sparse matrix as the observed data ponts.
        
        Returns:
            The Array like or sparse matrix of reconstructed instances of time-series.
        '''
        encoder_outputs = self.encoder.inference(time_series_X_arr)
        feature_points_arr = self.encoder.get_feature_points()
        feature_points_arr = feature_points_arr.T
        feature_points_arr = feature_points_arr[::-1]
        feature_points_arr = feature_points_arr.reshape((1, feature_points_arr.shape[0]))

        decoder_outputs = self.decoder.inference(feature_points_arr)
        reconstructed_arr = self.decoder.get_feature_points().T

        for _ in range(time_series_X_arr.shape[0]):
            decoder_outputs = decoder_outputs.reshape((1, decoder_outputs.shape[0]))
            decoder_outputs = self.decoder.inference(decoder_outputs)
            reconstructed_arr = np.r_[reconstructed_arr, self.decoder.get_feature_points().T]

        return reconstructed_arr

    def get_reconstruction_error(self):
        '''
        Extract the reconstruction errors as anomaly scores.

        Returns:
            Array like or sparse matrix of feature points or virtual visible observed data points.
        '''
        return np.array(self.__reconstructed_error_list)

    def set_readonly(self, value):
        raise TypeError("This property must be read-only.")

    def get_encoder(self):
        ''' getter '''
        return self.__encoder
    
    def set_encoder(self, value):
        ''' setter '''
        if isinstance(value, ReconstructableFeature):
            self.__encoder = value
        else:
            raise TypeError()

    encoder = property(get_encoder, set_encoder)

    def get_decoder(self):
        ''' getter '''
        return self.__decoder
    
    def set_decoder(self, value):
        ''' setter '''
        if isinstance(value, ReconstructableFeature):
            self.__decoder = value
        else:
            raise TypeError()

    decoder = property(get_decoder, set_decoder)

    def get_reconstruction_error_function(self):
        return self.__reconstruction_error_function

    reconstruction_error_function = property(get_reconstruction_error_function, set_readonly)

    def get_labeled_arr(self):
        ''' getter '''
        return np.array(self.__labeled_list)
    
    labeled_arr = property(get_labeled_arr, set_readonly)
