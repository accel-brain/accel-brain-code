# -*- coding: utf-8 -*-
import numpy as np
import torch

import pandas as pd
from logging import getLogger
import os

from accelbrainbase.extractabledata.labeled_csv_extractor import LabeledCSVExtractor
from accelbrainbase.iteratabledata.labeled_image_iterator import LabeledImageIterator as _LabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class LabeledCSVIterator(_LabeledImageIterator):
    '''
    Iterator that draws from CSV files and generates `mxnet.ndarray` of labeled samples.
    '''

    def __init__(
        self,
        labeled_csv_extractor,
        train_csv_path,
        test_csv_path,
        epochs=300,
        batch_size=20,
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
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            scale:                          `float` of scaling factor for data.
            noiseable_data:                 is-a `NoiseableData` for Denoising Auto-Encoders.
        '''
        if isinstance(labeled_csv_extractor, LabeledCSVExtractor) is False:
            raise TypeError("The type of `labeled_csv_extractor` must be `LabeledCSVExtractor`.")
        if noiseable_data is not None and isinstance(noiseable_data, NoiseableData) is False:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        train_observed_arr, train_label_arr = labeled_csv_extractor.extract(train_csv_path)
        test_observed_arr, test_label_arr = labeled_csv_extractor.extract(test_csv_path)

        dataset_size = train_observed_arr.shape[0]
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        train_label_df = pd.DataFrame(train_label_arr, columns=["tmp"])
        self.__label_n = train_label_df.tmp.drop_duplicates().values.shape[0]

        self.__train_observed_arr = train_observed_arr
        self.__train_label_arr = train_label_arr
        self.__test_observed_arr = test_observed_arr
        self.__test_label_arr = test_label_arr

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
            training_label_arr, test_label_arr = None, None
            row_arr = np.arange(self.__train_observed_arr.shape[0])
            np.random.shuffle(row_arr)

            training_batch_arr = self.__train_observed_arr[row_arr[:self.batch_size]]
            training_batch_arr = torch.from_numpy(training_batch_arr)
            training_batch_arr = training_batch_arr.to(self.__ctx).float()
            training_batch_arr = self.pre_normalize(training_batch_arr)

            label_key_arr = self.__train_label_arr[row_arr[:self.batch_size]]
            label_key_arr = torch.from_numpy(label_key_arr)
            label_key_arr = label_key_arr.to(self.__ctx).float()
            training_label_arr = torch.one_hot(label_key_arr, self.__label_n)

            test_row_arr = np.arange(self.__test_observed_arr.shape[0])
            np.random.shuffle(test_row_arr)

            test_batch_arr = self.__test_observed_arr[test_row_arr[:self.batch_size]]
            test_batch_arr = torch.from_numpy(test_batch_arr)
            test_batch_arr = test_batch_arr.to(self.__ctx).float()
            test_batch_arr = self.pre_normalize(test_batch_arr)

            test_label_key_arr = self.__test_label_arr[test_row_arr[:self.batch_size]]
            test_label_key_arr = torch.from_numpy(test_label_key_arr)
            test_label_key_arr = test_label_key_arr.to(self.__ctx).float()
            test_label_arr = torch.one_hot(test_label_key_arr, self.__label_n)

            if self.__noiseable_data is not None:
                training_batch_arr = self.__noiseable_data.noise(training_batch_arr)

            yield training_batch_arr, training_label_arr, test_batch_arr, test_label_arr

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
            yield None, None, test_batch_arr, None
