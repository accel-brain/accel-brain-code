# -*- coding: utf-8 -*-
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
from logging import getLogger
import os

from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata.unlabeled_image_iterator import UnlabeledImageIterator as _UnlabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class UnlabeledTHotTXTIterator(_UnlabeledImageIterator):
    '''
    Iterator that draws from CSV files and generates `mxnet.ndarray` of unlabeled samples.
    '''

    def get_token_list(self):
        return self.__token_list
    
    def set_token_list(self, value):
        self.__token_list = value
    
    token_list = property(get_token_list, set_token_list)

    def get_token_arr(self):
        return np.array(self.__token_list)
    
    def set_token_arr(self, value):
        raise TypeError("This property must be read-only.")
    
    token_arr = property(get_token_arr, set_token_arr)

    def get_pre_txt_arr(self):
        return self.__pre_txt_arr
    
    def set_pre_txt_arr(self, value):
        self.__pre_txt_arr = value

    pre_txt_arr = property(get_pre_txt_arr, set_pre_txt_arr)

    def __init__(
        self,
        train_txt_path_list,
        test_txt_path_list=None,
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
            train_txt_path_list:            `list` of `str` of path to CSV file in training.
            test_txt_path_list:             `list` of `str` of path to CSV file in test.

            epochs:                         `int` of epochs of Mini-batch.
            bath_size:                      `int` of batch size of Mini-batch.
            seq_len:                        `int` of the length of series. 
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            scale:                          `float` of scaling factor for data.
            noiseable_data:                 is-a `NoiseableData` for Denoising Auto-Encoders.
        '''
        if noiseable_data is not None and isinstance(noiseable_data, NoiseableData) is False:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        train_txt = ""
        for train_txt_path in train_txt_path_list:
            with open(train_txt_path) as f:
                train_txt += f.read()

        train_token_list = list(train_txt)
        self.__token_list = list(set(train_token_list))

        self.__train_txt_path_list = train_txt_path_list
        if test_txt_path_list is not None:
            self.__test_txt_path_list = test_txt_path_list
        else:
            self.__test_txt_path_list = train_txt_path_list

        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.norm_mode = norm_mode
        self.scale = scale
        self.__noiseable_data = noiseable_data

        self.__ctx = ctx

        self.__pre_txt_arr = None

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
            training_batch_arr = np.zeros(
                (
                    self.batch_size,
                    1,
                    self.seq_len,
                    len(self.__token_list)
                ),
            )
            test_batch_arr = np.zeros(
                (
                    self.batch_size,
                    1,
                    self.seq_len,
                    len(self.__token_list)
                ),
            )

            for i in range(self.batch_size):
                file_key = np.random.randint(low=0, high=len(self.__train_txt_path_list))
                with open(self.__train_txt_path_list[file_key]) as f:
                    train_txt = f.read()

                test_file_key = np.random.randint(low=0, high=len(self.__test_txt_path_list))
                with open(self.__test_txt_path_list[test_file_key]) as f:
                    test_txt = f.read()
                
                if self.__pre_txt_arr is None:
                    start_row = np.random.randint(low=0, high=len(train_txt) - self.seq_len)
                    test_start_row = np.random.randint(low=0, high=len(test_txt) - self.seq_len)

                    train_txt = train_txt[start_row:start_row+self.seq_len]
                    test_txt = test_txt[test_start_row:test_start_row+self.seq_len]
                else:
                    if train_txt.index(self.__pre_txt_arr[i]) + (self.seq_len * 2) < len(train_txt):
                        start_row = train_txt.index(self.__pre_txt_arr[i]) + self.seq_len
                    else:
                        start_row = np.random.randint(low=0, high=len(train_txt) - self.seq_len)
                    train_txt = train_txt[start_row:start_row+self.seq_len]

                    test_start_row = np.random.randint(low=0, high=len(test_txt) - self.seq_len)
                    test_txt = test_txt[test_start_row:test_start_row+self.seq_len]


                for seq in range(self.seq_len):
                    training_batch_arr[i, 0, seq, self.__token_list.index(train_txt[seq])] = 1.0
                    test_batch_arr[i, 0, seq, self.__token_list.index(test_txt[seq])] = 1.0

            training_batch_arr = nd.ndarray.array(training_batch_arr, ctx=self.__ctx)
            test_batch_arr = nd.ndarray.array(test_batch_arr, ctx=self.__ctx)

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
        for test_file_key in range(len(self.__test_txt_path_list)):
            global_seq = 0
            with open(self.__test_txt_path_list[test_file_key]) as f:
                test_txt = f.read()

            _test_batch_arr = np.zeros(
                (
                    self.batch_size,
                    1,
                    self.seq_len,
                    len(self.__token_list)
                ),
            )
            batch = 0
            for start_row in range(len(test_txt) - (self.seq_len)):
                for seq in range(self.seq_len):
                    _test_batch_arr[
                        batch, 
                        0, 
                        seq, 
                        self.__token_list.index(test_txt[start_row+seq])
                    ] = 1.0
                batch += 1
                if batch == self.batch_size:
                    test_batch_arr = nd.ndarray.array(_test_batch_arr, ctx=self.__ctx)
                    test_batch_arr = self.pre_normalize(test_batch_arr)
                    yield None, None, test_batch_arr, None
                    batch = 0
                    _test_batch_arr = np.zeros(
                        (
                            self.batch_size,
                            1,
                            self.seq_len,
                            len(self.__token_list)
                        ),
                    )
