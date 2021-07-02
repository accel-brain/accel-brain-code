# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from logging import getLogger
import os

from accelbrainbase.iteratabledata.transformer_iterator import TransformerIterator as _TransformerIterator
from accelbrainbase.noiseable_data import NoiseableData


class GaussTransformerIterator(_TransformerIterator):
    '''
    Iterator that draws from Gauss distribution.
    '''

    def __init__(
        self,
        loc=0.5,
        std=0.25,
        seq_len=5,
        dim=100,
        epochs=300,
        batch_size=20,
        norm_mode="z_score",
        noiseable_data=None,
        scale=1.0,
        ctx=mx.gpu(),
        dataset_size=1000
    ):
        '''
        Init.

        Args:
            loc:                            Mean (“centre”) of the distribution.
            std:                            Scale of the distribution.
            seq_len:                        The length of sequence.
            dim:                            Dimension.
            epochs:                         `int` of epochs of Mini-batch.
            batch_size:                      `int` of batch size of Mini-batch.
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            noiseable_data:                 is-a `NoiseableData` for Denoising Auto-Encoders.
            dataset_size:                   `int` of the dataset size.
        '''
        if noiseable_data is not None and isinstance(noiseable_data, NoiseableData) is False:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.__loc = loc
        self.__std = std
        self.__seq_len = seq_len
        self.__dim = dim

        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.norm_mode = norm_mode
        self.scale = scale
        self.__noiseable_data = noiseable_data

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
        '''
        for _ in range(self.iter_n):
            training_encoded_observed_arr, test_encoded_observed_arr = None, None
            training_decoded_observed_arr, test_decoded_observed_arr = None, None

            training_encoded_observed_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__seq_len,
                    self.__dim
                ),
                ctx=self.__ctx
            )
            training_encoded_mask_arr = None

            training_decoded_observed_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__seq_len,
                    self.__dim
                ),
                ctx=self.__ctx
            )
            training_decoded_mask_arr = None

            test_encoded_observed_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__seq_len,
                    self.__dim
                ),
                ctx=self.__ctx
            )
            test_encoded_mask_arr = None

            test_decoded_observed_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__seq_len,
                    self.__dim
                ),
                ctx=self.__ctx
            )
            test_decoded_mask_arr = None

            training_encoded_observed_arr = self.pre_normalize(training_encoded_observed_arr)
            training_decoded_observed_arr = self.pre_normalize(training_decoded_observed_arr)
            test_encoded_observed_arr = self.pre_normalize(test_encoded_observed_arr)
            test_decoded_observed_arr = self.pre_normalize(test_decoded_observed_arr)

            if self.__noiseable_data is not None:
                training_encoded_observed_arr = self.__noiseable_data.noise(training_encoded_observed_arr)
                training_decoded_observed_arr = self.__noiseable_data.noise(training_decoded_observed_arr)

            yield training_encoded_observed_arr, training_decoded_observed_arr, training_encoded_mask_arr, training_decoded_mask_arr, test_encoded_observed_arr, test_decoded_observed_arr, test_encoded_mask_arr, test_decoded_mask_arr

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
