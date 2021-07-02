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


class TokenAttentionAutoEncoderIterator(_TransformerIterator, _TokenIterator):
    '''
    Iterator that draws from Gauss distribution.
    '''

    def __init__(
        self, 
        seq_token_list,
        vectorizable_token, 
        test_vectorizable_token=None,
        epochs=1000,
        batch_size=25,
        seq_len=5,
        test_size=0.3,
        norm_mode=None,
        noiseable_data=None,
        scale=1.0,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            seq_token_list:         `list` of `str` of tokens.
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
        self.vectorizable_token = vectorizable_token
        if test_vectorizable_token is not None:
            self.test_vectorizable_token = test_vectorizable_token
        else:
            self.test_vectorizable_token = vectorizable_token

        dataset_size = len(seq_token_list) / seq_len
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.seq_token_list = seq_token_list
        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.__norm_mode = norm_mode
        self.__scale = scale
        self.__noiseable_data = noiseable_data
        self.__ctx = ctx

    def generate_learned_samples(self):
        '''
        Draw and generate data.

        Returns:
            `Tuple` data. The shape is ...
            - observed data points in training.
            - masked data points in training.
            - observed data points in test.
            - masked data points in test.
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
            test_encoded_observed_arr = np.empty(
                (
                    self.batch_size,
                    self.seq_len,
                    self.vectorizable_token.dim
                )
            )
            for batch in range(self.batch_size):
                for _ in range(5):
                    try:
                        key_arr = np.random.randint(low=0, high=self.vectorizable_token.dim, size=self.seq_len)
                        training_encoded_observed_arr[batch] = self.vectorizable_token.vectorize(
                            self.vectorizable_token.token_arr[key_arr].tolist()
                        )
                        break
                    except IndexError:
                        continue

                for _ in range(5):
                    try:
                        key_arr = np.random.randint(low=0, high=self.test_vectorizable_token.dim, size=self.seq_len)
                        test_encoded_observed_arr[batch] = self.test_vectorizable_token.vectorize(
                            self.test_vectorizable_token.token_arr[key_arr].tolist()
                        )

                        break
                    except IndexError:
                        continue

            training_encoded_observed_arr = self.pre_normalize(training_encoded_observed_arr)
            test_encoded_observed_arr = self.pre_normalize(test_encoded_observed_arr)

            training_encoded_observed_arr = nd.ndarray.array(
                training_encoded_observed_arr,
                ctx=self.__ctx
            )
            training_target_arr = training_encoded_observed_arr.copy()

            test_encoded_observed_arr = nd.ndarray.array(
                test_encoded_observed_arr,
                ctx=self.__ctx
            )
            test_target_arr = test_encoded_observed_arr.copy()

            if self.__noiseable_data is not None:
                training_encoded_observed_arr = self.__noiseable_data.noise(training_encoded_observed_arr)
                training_decoded_observed_arr = self.__noiseable_data.noise(training_decoded_observed_arr)

            yield training_encoded_observed_arr, None, None, test_encoded_observed_arr, None, None, training_target_arr, test_target_arr

    def generate_samples_and_noises(self, test_mode=True):
        '''
        Draw and generate data and noised data.
        The targets will be drawn from all image file sorted in ascending order by file name.

        Returns:
            `Tuple` data. The shape is ...
            - encoder's observed data points in training.
            - decoder's observed data points in training.
            - encoder's masked data points in training.
            - decoder's masked data points in training.
            - `list` of tokens.
        '''
        if test_mode is False:
            vectorizable_token = self.test_vectorizable_token
        else:
            vectorizable_token = self.vectorizable_token

        encoded_observed_arr = np.empty(
            (
                self.batch_size,
                self.seq_len,
                self.vectorizable_token.dim
            )
        )
        decoded_observed_arr = np.empty(
            (
                self.batch_size,
                self.seq_len,
                self.vectorizable_token.dim
            )
        )

        batch = 0
        _token_list = []
        for i in range(self.seq_len, vectorizable_token.dim):
            key_arr = np.array(list(range(i-self.seq_len, i)))
            token_list = vectorizable_token.token_arr[key_arr].tolist()
            decoded_observed_arr[batch] = vectorizable_token.vectorize(
                token_list
            )
            _token_list.append("<sep>".join(token_list))
            if batch == self.batch_size - 1:
                decoded_observed_arr = nd.ndarray.array(
                    decoded_observed_arr,
                    ctx=self.__ctx
                )
                encoded_observed_arr = nd.random.uniform_like(
                    decoded_observed_arr
                )
                batch = 0
                yield encoded_observed_arr, decoded_observed_arr, None, None, _token_list
                _token_list = []

            batch += 1

    def generate_inferenced_samples(self, media_token_list, test_mode=True):
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
        if test_mode is False:
            vectorizable_token = self.test_vectorizable_token
        else:
            vectorizable_token = self.vectorizable_token

        encoded_observed_arr = np.empty(
            (
                self.batch_size,
                self.seq_len,
                self.vectorizable_token.dim
            )
        )
        decoded_observed_arr = np.empty(
            (
                self.batch_size,
                self.seq_len,
                self.vectorizable_token.dim
            )
        )

        batch = 0
        _token_list = []
        for i in range(self.seq_len, vectorizable_token.dim):
            key_arr = np.array(list(range(i-self.seq_len, i)))
            token_list = vectorizable_token.token_arr[key_arr].tolist()
            decoded_observed_arr[batch] = vectorizable_token.vectorize(
                token_list
            )
            _token_list.append("<sep>".join(token_list))
            encoded_observed_arr[batch] = vectorizable_token.vectorize(
                media_token_list
            )
            if batch == self.batch_size - 1:
                decoded_observed_arr = nd.ndarray.array(
                    decoded_observed_arr,
                    ctx=self.__ctx
                )
                encoded_observed_arr = nd.ndarray.array(
                    encoded_observed_arr,
                    ctx=self.__ctx
                )
                batch = 0
                yield encoded_observed_arr, decoded_observed_arr, None, None, _token_list
                _token_list = []

            batch += 1

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
