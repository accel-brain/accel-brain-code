# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from logging import getLogger
import os

from accelbrainbase.iteratabledata.transformer_iterator import TransformerIterator as _TransformerIterator
from accelbrainbase.noiseable_data import NoiseableData
from pysummarization.iteratabledata.token_iterator import TokenIterator as _TokenIterator


class BiSentenceTransformerIterator(_TransformerIterator, _TokenIterator):
    '''
    Iterator that draws from Gauss distribution.
    '''

    __label_smoothing_weight = 0.01

    def get_label_smoothing_weight(self):
        ''' getter '''
        return self.__label_smoothing_weight
    
    def set_label_smoothing_weight(self, value):
        ''' setter '''
        self.__label_smoothing_weight = value

    label_smoothing_weight = property(get_label_smoothing_weight, set_label_smoothing_weight)

    def __init__(
        self, 
        sentence_list,
        vectorizable_token, 
        epochs=1000,
        batch_size=25,
        seq_len=5,
        test_size=0.3,
        norm_mode=None,
        noiseable_data=None,
        label_smoothing_alpha=0.01,
        scale=1.0,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            sentence_list:          [[token, token, ...], ...]
            vectorizable_token:     is-a `VectorizableToken`.
            epochs:                 `int` of epochs.
            batch_size:             `int` of batch size.
            seq_len:                `int` of length of series.
            test_size:              `float` of rate of test data.
                                    training data : test data = (1 - test_size) : test_size

            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            noiseable_data:         is-a `NoiseableData`.
        '''
        _sentence_list = []
        for token_list in sentence_list:
            if len(token_list) > (seq_len * 3) + 1:
                _sentence_list.append(token_list)

        sentence_list = _sentence_list

        self.vectorizable_token = vectorizable_token

        self.sentence_arr = np.array(sentence_list)

        training_row = int(len(self.sentence_arr) * (1 - test_size))
        key_arr = np.arange(len(self.sentence_arr))
        np.random.shuffle(key_arr)
        self.training_sentence_arr = self.sentence_arr[key_arr[:training_row]]
        self.test_sentence_arr = self.sentence_arr[key_arr[training_row:]]

        dataset_size = self.training_sentence_arr.shape[0]
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.__norm_mode = norm_mode
        self.__scale = scale
        self.__noiseable_data = noiseable_data
        self.__label_smoothing_alpha = label_smoothing_alpha
        self.__ctx = ctx

    def generate_learned_samples(self):
        '''
        Draw and generate data.

        Returns:
            `Tuple` data. The shape is ...
            - encoder's observed data points in training.
            - decoder's observed data points in training.
            - encoder's masked data points in training.
            - decoder's masked data points in training.
            - encoder's observed data points in test.
            - decoder's observed data points in test.
            - encoder's masked data points in test.
            - decoder's masked data points in test.
            - training target data.
            - test target data.
        '''
        for _ in range(self.iter_n):
            training_encoded_observed_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            training_decoded_observed_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            training_target_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            test_encoded_observed_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            test_decoded_observed_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            test_target_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )

            for batch in range(self.batch_size):
                for _ in range(5):
                    try:
                        key = np.random.randint(low=0, high=len(self.training_sentence_arr))
                        start = np.random.randint(low=0, high=len(self.training_sentence_arr[key]) - (self.seq_len * 3) - 1)
                        end = start + self.seq_len
                        token_list = self.training_sentence_arr[key][start:end]
                        training_decoded_observed_arr[batch] = self.vectorizable_token.vectorize(token_list)

                        token_list = self.training_sentence_arr[key][end:end+self.seq_len]
                        training_encoded_observed_arr[batch] = self.vectorizable_token.vectorize(token_list)

                        token_list = self.training_sentence_arr[key][end+self.seq_len:end+self.seq_len+self.seq_len]
                        training_target_arr[batch] = self.vectorizable_token.vectorize(token_list)

                        break
                    except IndexError:
                        continue

                for _ in range(5):
                    try:
                        key = np.random.randint(low=0, high=len(self.test_sentence_arr))
                        start = np.random.randint(low=0, high=len(self.test_sentence_arr[key]) - (self.seq_len * 3) - 1)
                        end = start + self.seq_len

                        token_list = self.test_sentence_arr[key][start:end]
                        test_decoded_observed_arr[batch] = self.vectorizable_token.vectorize(token_list)

                        token_list = self.test_sentence_arr[key][end:end+self.seq_len]
                        test_encoded_observed_arr[batch] = self.vectorizable_token.vectorize(token_list)

                        token_list = self.test_sentence_arr[key][end+self.seq_len:end+self.seq_len+self.seq_len]
                        test_target_arr[batch] = self.vectorizable_token.vectorize(token_list)
                        break
                    except IndexError:
                        continue

            training_encoded_observed_arr = self.pre_normalize(training_encoded_observed_arr)
            training_decoded_observed_arr = self.pre_normalize(training_decoded_observed_arr)
            test_encoded_observed_arr = self.pre_normalize(test_encoded_observed_arr)
            test_decoded_observed_arr = self.pre_normalize(test_decoded_observed_arr)

            training_encoded_observed_arr = nd.ndarray.array(
                training_encoded_observed_arr,
                ctx=self.__ctx
            )
            training_decoded_observed_arr = nd.ndarray.array(
                training_decoded_observed_arr,
                ctx=self.__ctx
            )
            training_target_arr = nd.ndarray.array(
                training_target_arr,
                ctx=self.__ctx
            )
            test_encoded_observed_arr = nd.ndarray.array(
                test_encoded_observed_arr,
                ctx=self.__ctx
            )
            test_decoded_observed_arr = nd.ndarray.array(
                test_decoded_observed_arr,
                ctx=self.__ctx
            )
            test_target_arr = nd.ndarray.array(
                test_target_arr,
                ctx=self.__ctx
            )

            if self.__noiseable_data is not None:
                training_encoded_observed_arr = self.__noiseable_data.noise(training_encoded_observed_arr)
                training_decoded_observed_arr = self.__noiseable_data.noise(training_decoded_observed_arr)

            training_target_arr = self.label_smoothing(training_target_arr)
            test_target_arr = self.label_smoothing(test_target_arr)

            yield training_encoded_observed_arr, training_decoded_observed_arr, None, None, test_encoded_observed_arr, test_decoded_observed_arr, None, None, training_target_arr, test_target_arr

    def generate_inferenced_samples(self):
        '''
        Draw and generate data.
        The targets will be drawn from all image file sorted in ascending order by file name.

        Returns:
            `Tuple` data. The shape is ...
            - `None`.
            - `None`.
            - `mxnet.ndarray` of observed data points in test.
            - file path.
        '''
        raise NotImplementedError()

    def pre_normalize(self, arr):
        '''
        Normalize before observation.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        if self.__norm_mode is None:
            return arr
        
        return super().pre_normalize(arr)

    def label_smoothing(self, arr):
        arr = arr * (1 - self.__label_smoothing_alpha) + (self.__label_smoothing_alpha * (1 / arr.shape[2]))
        return arr
