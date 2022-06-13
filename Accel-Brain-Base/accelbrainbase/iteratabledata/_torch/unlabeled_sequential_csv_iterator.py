# -*- coding: utf-8 -*-
import numpy as np

import torch

import pandas as pd
from logging import getLogger
import os

from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata.labeled_image_iterator import LabeledImageIterator as _LabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class UnlabeledSequentialCSVIterator(_LabeledImageIterator):
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
        seq_len=10,
        norm_mode="z_score",
        scale=1.0,
        noiseable_data=None,
        ctx="cpu"
    ):
        '''
        Init.

        Args:
            image_extractor:                is-a `ImageExtractor`.
            train_csv_path:                 `str` of path to CSV file in training.
            test_csv_path:                  `str` of path to CSV file in test.

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

        train_observed_arr = unlabeled_csv_extractor.extract(train_csv_path)
        test_observed_arr = unlabeled_csv_extractor.extract(test_csv_path)

        dataset_size = train_observed_arr.shape[0]
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.__train_observed_arr = train_observed_arr
        self.__test_observed_arr = test_observed_arr

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

            row_arr = np.arange(self.__train_observed_arr.shape[0] - self.seq_len)
            np.random.shuffle(row_arr)
            test_row_arr = np.arange(self.__test_observed_arr.shape[0] - self.seq_len)
            np.random.shuffle(test_row_arr)

            training_batch_arr = None
            test_batch_arr = None
            for batch in range(self.batch_size):
                arr = self.__train_observed_arr[row_arr[batch]:row_arr[batch]+self.seq_len]
                arr = torch.from_numpy(arr)
                arr = arr.to(self.__ctx).float()
                arr = torch.unsqueeze(
                    arr,
                    dim=0
                )
                if training_batch_arr is None:
                    training_batch_arr = arr
                else:
                    training_batch_arr = torch.cat(
                        (
                            training_batch_arr,
                            arr,
                        ),
                        dim=0
                    )

                test_arr = self.__test_observed_arr[test_row_arr[batch]:test_row_arr[batch]+self.seq_len]
                test_arr = torch.from_numpy(test_arr)
                test_arr = test_arr.to(self.__ctx).float()
                test_arr = torch.unsqueeze(
                    test_arr,
                    dim=0
                )
                if test_batch_arr is None:
                    test_batch_arr = test_arr
                else:
                    test_batch_arr = torch.cat(
                        (
                            test_batch_arr,
                            test_arr,
                        ),
                        dim=0
                    )

            training_batch_arr = self.pre_normalize(training_batch_arr)
            test_batch_arr = self.pre_normalize(test_batch_arr)

            if self.__noiseable_data is not None:
                training_batch_arr = self.__noiseable_data.noise(training_batch_arr)
                test_batch_arr = self.__noiseable_data.noise(test_batch_arr)

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
        for row in range(self.seq_len, self.__test_observed_arr.shape[0]):
            test_arr = self.__test_observed_arr[row-self.seq_len:row]
            test_arr = torch.from_numpy(test_arr)
            test_arr = test_arr.to(self.__ctx).float()
            test_arr = torch.unsqueeze(
                test_arr,
                dim=0
            )
            if test_batch_arr is None:
                test_batch_arr = test_arr
            else:
                test_batch_arr = torch.cat(
                    (
                        test_batch_arr,
                        test_arr,
                    ),
                    dim=0
                )
            if test_batch_arr.shape[0] == self.batch_size:
                yield None, None, test_batch_arr, None
                test_batch_arr = None

    def pre_normalize(self, arr):
        '''
        Normalize before observation.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        if self.norm_mode == "min_max":
            if torch.max(arr) != torch.min(arr):
                n = 0.0
            else:
                n = 1e-08
            arr = (arr - torch.min(arr)) / (torch.max(arr) - torch.min(arr) + n)
        elif self.norm_mode == "z_score":
            std = torch.std(arr)
            if std == 0:
                std += 1e-08
            arr = (arr - torch.mean(arr)) / std

        arr = arr * self.scale
        return arr
