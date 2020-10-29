# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from logging import getLogger
import os

from accelbrainbase.iteratabledata.labeled_image_iterator import LabeledImageIterator as _LabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class GaussIterator(_LabeledImageIterator):
    '''
    Iterator that draws from CSV files and generates `mxnet.ndarray`.
    '''

    def __init__(
        self,
        loc=0.0,
        std=1.0,
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
            image_extractor:                is-a `ImageExtractor`.
            train_csv_path:                 `str` of path to CSV file in training.
            test_csv_path:                  `str` of path to CSV file in test.

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
            - `mxnet.ndarray` of observed data points in training.
            - `mxnet.ndarray` of supervised data in training.
            - `mxnet.ndarray` of observed data points in test.
            - `mxnet.ndarray` of supervised data in test.
        '''
        for _ in range(self.iter_n):
            training_batch_arr, test_batch_arr = None, None

            training_batch_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__dim
                ),
                ctx=self.__ctx
            )
            test_batch_arr = nd.random.normal(
                loc=self.__loc,
                scale=self.__std,
                shape=(
                    self.batch_size,
                    self.__dim
                ),
                ctx=self.__ctx
            )

            training_batch_arr = self.pre_normalize(training_batch_arr)
            test_batch_arr = self.pre_normalize(test_batch_arr)

            if self.__noiseable_data is not None:
                training_batch_arr = self.__noiseable_data.noise(training_batch_arr)

            yield training_batch_arr, training_batch_arr, test_batch_arr, test_batch_arr

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
