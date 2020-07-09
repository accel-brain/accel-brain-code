# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from logging import getLogger
import os

from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata.labeled_image_iterator import LabeledImageIterator as _LabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class UnlabeledCSVIterator(_LabeledImageIterator):
    '''
    Iterator that draws from CSV files and generates `mxnet.ndarray` of unlabeled samples.
    '''

    def __init__(
        self,
        unlabeled_csv_extractor,
        train_csv_path,
        test_csv_path,
        epochs=300,
        batch_size=20,
        norm_mode="z_score",
        scale=1.0,
        noiseable_data=None,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            image_extractor:                is-a `ImageExtractor`.
            train_csv_path:                 `str` of path to CSV file in training.
            test_csv_path:                  `str` of path to CSV file in test.

            epochs:                         `int` of epochs of Mini-batch.
            bath_size:                      `int` of batch size of Mini-batch.
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            scale:                          `float` of scaling factor for data.
            noiseable_data:                 is-a `NoiseableData` for Denoising Auto-Encoders.
        '''
        if isinstance(unlabeled_csv_extractor, UnlabeledCSVExtractor) is False:
            raise TypeError("The type of `unlabeled_csv_extractor` must be `UnlabeledCSVExtractor`.")
        if noiseable_data is not None and isinstance(noiseable_data, NoiseableData) is False:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        train_observed_arr = unlabeled_csv_extractor.extract(train_csv_path)
        test_observed_arr = unlabeled_csv_extractor.extract(test_csv_path)

        self.__train_observed_arr = train_observed_arr
        self.__test_observed_arr = test_observed_arr

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
        for epoch in range(self.epochs):
            training_batch_arr, test_batch_arr = None, None

            row_arr = np.arange(self.__train_observed_arr.shape[0])
            np.random.shuffle(row_arr)

            training_batch_arr = self.__train_observed_arr[row_arr[:self.batch_size]]
            training_batch_arr = mx.ndarray.array(training_batch_arr, ctx=self.__ctx)
            training_batch_arr = self.pre_normalize(training_batch_arr)

            test_row_arr = np.arange(self.__test_observed_arr.shape[0])
            np.random.shuffle(test_row_arr)

            test_batch_arr = self.__test_observed_arr[test_row_arr[:self.batch_size]]
            test_batch_arr = mx.ndarray.array(test_batch_arr, ctx=self.__ctx)
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
        for i in range(1, self.__test_observed_arr.shape[0] // self.batch_size):
            test_batch_arr = self.__test_observed_arr[(i-1)*self.batch_size:i*self.batch_size]
            test_batch_arr = mx.ndarray.array(test_batch_arr, ctx=self.__ctx)
            yield None, None, test_batch_arr, None
