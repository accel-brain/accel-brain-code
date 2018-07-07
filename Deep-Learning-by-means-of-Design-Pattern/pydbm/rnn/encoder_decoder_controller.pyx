# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss
from pydbm.rnn.interface.reconstructable_feature import ReconstructableFeature


class EncoderDecoderControler(object):
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
        computable_loss,
        test_size_rate=0.3
    ):
        '''
        Init.

        Args:
            encoder:    is-a `ReconstructableFeature` for vector representation of the input time-series.
            decoder:    is-a `ReconstructableFeature` to reconstruct the time-series.

            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            learning_rate:                  Learning rate.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            computable_loss:                Loss function for training Encoder/Decoder.

            test_size_rate:                 Size of Test data set. If this value is `0`, the validation will not be executed.

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

        self.__computable_loss = computable_loss

        logger = getLogger("pydbm.encoder_decoder_controller")
        self.__logger = logger
        self.__logger.debug("encoder_decoder_controller is started. ")

    def learn(self, observed_arr, target_arr=None):
        '''
        Learn the observed data points.
        for vector representation of the input time-series.

        In Encoder-Decode scheme, usecase of this method may be pre-training with `__learned_params_dict`.

        Override.

        Args:
            image_generator:    is-a `ImageGenerator`.

        '''
        self.__logger.debug("encoder_decoder_controler.learn is started. ")
        # to reduce gradually using `__learning_attenuate_rate`.
        learning_rate = self.__learning_rate

        if target_arr is not None:
            if target_arr.shape[0] != observed_arr.shape[0]:
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

            input_arr = None
            test_input_arr = None
            reconstructed_arr = None
            test_reconstructed_arr = None
            train_labeled = None
            test_labeled = None

            if epoch == 0:
                self.__logger.debug("batch is started.")

            for row in range(train_observed_arr.shape[0]):
                if epoch == 0:
                    self.__logger.debug("Generation image is end.")

                if input_arr is None:
                    input_arr = train_observed_arr[row]
                else:
                    input_arr = np.r_[input_arr, train_observed_arr[row]]

                if epoch == 0:
                    self.__logger.debug("`input_arr` is set.")

                encoder_outputs = self.encoder.inference(train_observed_arr)
                if epoch == 0:
                    self.__logger.debug("`encoder_outputs` is set.")

                feature_points_arr = self.encoder.get_feature_points()
                feature_points_arr = feature_points_arr.T
                feature_points_arr = feature_points_arr[::-1]
                feature_points_arr = feature_points_arr.reshape((1, feature_points_arr.shape[0], feature_points_arr.shape[1]))
                if epoch == 0:
                    self.__logger.debug("`feature_points_arr` is set.")

                if epoch == 0:
                    self.__logger.debug("Gradients have been attached.")

                decoder_outputs = self.decoder.inference(feature_points_arr)
                if epoch == 0:
                    self.__logger.debug("`decoder_outputs` is set.")

                if reconstructed_arr is None:
                    reconstructed_arr = self.decoder.get_feature_points().T
                else:
                    reconstructed_arr = np.r_[reconstructed_arr, self.decoder.get_feature_points().T]

                if epoch == 0:
                    self.__logger.debug("`reconstructed_arr` is set.")

                train_labeled = train_target_arr[0][0]
                if self.__test_size_rate > 0:
                    test_labeled = test_target_arr[0][0]

                if self.__test_size_rate > 0:
                    if epoch == 0:
                        self.__logger.debug("validation is started.")

                    if test_input_arr is None:
                        test_input_arr = test_observed_arr
                    else:
                        test_input_arr = np.r_[test_input_arr, test_observed_arr]

                    if test_reconstructed_arr is None:
                        if epoch == 0:
                            self.__logger.debug("Infernecing is started.")
                        test_reconstructed_arr = self.inference(test_observed_arr).T
                        if epoch == 0:
                            self.__logger.debug("Infernecing is end.")
                    else:
                        if epoch == 0:
                            self.__logger.debug("Infernecing is started.")
                        test_reconstructed_arr = np.r_[
                            test_reconstructed_arr,
                            self.inference(test_observed_arr).T
                        ]
                        if epoch == 0:
                            self.__logger.debug("Infernecing is end.")
                    if epoch == 0:
                        self.__logger.debug("validation is end.")

                if epoch == 0:
                    self.__logger.debug("batch is end.")
                loss = self.computable_loss.compute(reconstructed_arr, input_arr[:][-1])
                test_loss = self.computable_loss.compute(test_reconstructed_arr, test_input_arr[:][-1])

            self.encoder.optimize(loss, learning_rate)
            self.decoder.optimize(loss, learning_rate)

            if epoch == 0:
                self.__logger.debug("Optimization is end.")

            reconstructed_error = self.computable_loss.compute(reconstructed_arr, input_arr[:][-1])
            if self.__test_size_rate > 0:
                test_reconstructed_error = self.computable_loss.compute(test_reconstructed_arr, test_input_arr[:][-1])

            if epoch == 0:
                self.__logger.debug("Computing reconstruction error is end.")

            # Keeping a moving average of the losses.
            if epoch == 0:
                moving_loss = np.mean(loss)
                moving_reconstructed_error = np.mean(reconstructed_error)
                if self.__test_size_rate > 0:
                    test_moving_loss = np.mean(test_loss)
                    test_moving_reconstructed_error = np.mean(test_reconstructed_error)

            else:
                moving_loss = .99 * moving_loss + .01 * np.mean(loss)
                moving_reconstructed_error = .99 * moving_reconstructed_error + .01 * np.mean(reconstructed_error)
                if self.__test_size_rate > 0:
                    test_moving_loss = .99 * test_moving_loss + .01 * np.mean(test_loss)
                    test_moving_reconstructed_error = .99 * test_moving_reconstructed_error + .01 * np.mean(test_reconstructed_error)

            if self.__test_size_rate > 0:
                self.__learned_result_list.append((epoch, moving_loss, moving_reconstructed_error, test_moving_loss, test_moving_reconstructed_error))
                self.__reconstructed_error_list.append((moving_reconstructed_error, test_moving_reconstructed_error))
                self.__labeled_list.append((train_labeled, test_labeled))
                self.__logger.debug("Epoch: " + str(epoch))
                self.__logger.debug("train loss:" + str(moving_loss))
                self.__logger.debug("train reconstruction error: " + str(moving_reconstructed_error))
                self.__logger.debug("test loss: " + str(test_moving_loss))
                self.__logger.debug("test reconstruction error: " + str(test_moving_reconstructed_error))
            else:
                self.__learned_result_list.append((epoch, moving_loss, moving_reconstructed_error))
                self.__reconstructed_error_list.append(moving_reconstructed_error)
                self.__labeled_list.append(train_labeled)
                self.__logger.debug("Epoch: " + str(epoch))
                self.__logger.debug("train loss:" + str(moving_loss))
                self.__logger.debug("reconstruction error: " + str(moving_reconstructed_error))

    def inference(self, time_series_X_arr):
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
        feature_points_arr = feature_points_arr[::-1].T
        feature_points_arr = feature_points_arr.reshape((1, feature_points_arr.shape[0], feature_points_arr.shape[1]))
        decoder_outputs = self.decoder.inference(feature_points_arr)
        reconstructed_arr = self.decoder.get_feature_points()
        reconstructed_arr = reconstructed_arr[::-1]
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
