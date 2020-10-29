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


class UnlabeledTHotCSVIterator(_LabeledImageIterator):
    '''
    Iterator that draws from CSV files and generates `mxnet.ndarray` of unlabeled samples.
    '''

    def __init__(
        self,
        unlabeled_csv_extractor,
        train_csv_path_list,
        test_csv_path_list,
        epochs=300,
        batch_size=20,
        seq_len=10,
        norm_mode="z_score",
        scale=1.0,
        noiseable_data=None,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            image_extractor:                is-a `ImageExtractor`.
            train_csv_path_list:            `list` of `str` of path to CSV file in training.
            test_csv_path_list:             `list` of `str` of path to CSV file in test.

            epochs:                         `int` of epochs of Mini-batch.
            batch_size:                     `int` of batch size of Mini-batch.
            seq_len:                        `int` of the length of series. 
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

        self.__unlabeled_csv_extractor = unlabeled_csv_extractor

        self.__train_csv_path_list = train_csv_path_list
        self.__test_csv_path_list = test_csv_path_list

        dataset_size = 0
        for file_key in range(len(self.__train_csv_path_list)):
            train_observed_arr = self.__unlabeled_csv_extractor.extract(
                self.__train_csv_path_list[file_key]
            )
            dataset_size = dataset_size + train_observed_arr.shape[0]

        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
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

            for i in range(self.batch_size):
                file_key = np.random.randint(low=0, high=len(self.__train_csv_path_list))
                train_observed_arr = self.__unlabeled_csv_extractor.extract(
                    self.__train_csv_path_list[file_key]
                )
                test_file_key = np.random.randint(low=0, high=len(self.__test_csv_path_list))
                test_observed_arr = self.__unlabeled_csv_extractor.extract(
                    self.__test_csv_path_list[test_file_key]
                )

                train_observed_arr = np.identity(
                    1 + int(train_observed_arr.max() + (train_observed_arr.min() * -1))
                )[
                    (train_observed_arr.reshape(train_observed_arr.shape[0], -1) + (train_observed_arr.min() * -1)).astype(int)
                ]

                test_observed_arr = np.identity(
                    1 + int(test_observed_arr.max() + (test_observed_arr.min() * -1))
                )[
                    (test_observed_arr.reshape(test_observed_arr.shape[0], -1) + (test_observed_arr.min() * -1)).astype(int)
                ]

                start_row = np.random.randint(low=0, high=train_observed_arr.shape[0] - self.seq_len)
                test_start_row = np.random.randint(low=0, high=test_observed_arr.shape[0] - self.seq_len)

                train_observed_arr = train_observed_arr[start_row:start_row+self.seq_len]
                test_observed_arr = test_observed_arr[test_start_row:test_start_row+self.seq_len]

                if training_batch_arr is None:
                    training_batch_arr = nd.expand_dims(
                        nd.ndarray.array(train_observed_arr, ctx=self.__ctx),
                        axis=0
                    )
                else:
                    training_batch_arr = nd.concat(
                        training_batch_arr,
                        nd.expand_dims(
                            nd.ndarray.array(train_observed_arr, ctx=self.__ctx),
                            axis=0
                        ),
                        dim=0
                    )

                if test_batch_arr is None:
                    test_batch_arr = nd.expand_dims(
                        nd.ndarray.array(test_observed_arr, ctx=self.__ctx),
                        axis=0
                    )
                else:
                    test_batch_arr = nd.concat(
                        test_batch_arr,
                        nd.expand_dims(
                            nd.ndarray.array(test_observed_arr, ctx=self.__ctx),
                            axis=0
                        ),
                        dim=0
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
        test_batch_arr = None

        for test_file_key in range(len(self.__test_csv_path_list)):
            test_observed_arr = self.__unlabeled_csv_extractor.extract(
                self.__test_csv_path_list[test_file_key]
            )
            test_observed_arr = np.identity(
                1 + int(test_observed_arr.max() + (test_observed_arr.min() * -1))
            )[
                (test_observed_arr.reshape(test_observed_arr.shape[0], -1) + (test_observed_arr.min() * -1)).astype(int)
            ]

            for row in range(self.seq_len, test_observed_arr.shape[0]):
                test_observed_arr = test_observed_arr[row-self.seq_len:row]

                test_arr = mx.ndarray.array(test_observed_arr, ctx=self.__ctx)
                test_arr = nd.expand_dims(
                    test_arr,
                    axis=0
                )
                if test_batch_arr is None:
                    test_batch_arr = test_arr
                else:
                    test_batch_arr = nd.concat(
                        test_batch_arr,
                        test_arr,
                        dim=0
                    )
                if test_batch_arr.shape[0] == self.batch_size:
                    yield None, None, test_batch_arr, None
                    test_batch_arr = None
