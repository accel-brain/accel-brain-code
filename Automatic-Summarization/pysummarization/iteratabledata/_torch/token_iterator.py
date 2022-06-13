# -*- coding: utf-8 -*-
from pysummarization.iteratabledata.token_iterator import TokenIterator as _TokenIterator
import torch
import numpy as np


class TokenIterator(_TokenIterator):
    '''
    Token Iterator.
    '''

    def __init__(
        self, 
        vectorizable_token, 
        token_arr, 
        epochs=1000,
        batch_size=25,
        seq_len=5,
        test_size=0.3,
        norm_mode="z_score",
        noiseable_data=None,
        ctx="cpu"
    ):
        '''
        Init.

        Args:
            vectorizable_token:     is-a `VectorizableToken`.
            token_arr:              `np.ndarray` of token vectors.
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
        super().__init__(
            vectorizable_token=vectorizable_token,
            token_arr=token_arr,
            epochs=epochs,
            batch_size=batch_size,
            seq_len=seq_len,
            test_size=test_size,
            norm_mode=norm_mode,
            noiseable_data=noiseable_data
        )
        self.__ctx = ctx
        self.__noiseable_data = noiseable_data

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
        for training_batch_arr, _, test_batch_arr, _ in super().generate_learned_samples():
            training_batch_arr = torch.from_numpy(
                training_batch_arr
            )
            training_batch_arr = training_batch_arr.to(self.__ctx)
            test_batch_arr = torch.from_numpy(test_batch_arr)
            test_batch_arr = test_batch_arr.to(self.__ctx)
            if self.__noiseable_data is not None:
                training_batch_arr = self.__noiseable_data.noise(training_batch_arr)

            yield training_batch_arr.float(), training_batch_arr.float().detach(), test_batch_arr.float(), test_batch_arr.float().detach()

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
        for _, _, test_batch_arr, _ in super().generate_inferenced_samples():
            test_batch_arr = torch.from_numpy(test_batch_arr)
            test_batch_arr = test_batch_arr.to(self.__ctx)
            yield None, None, test_batch_arr.float(), None

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
        elif self.norm_mode == "min_max":
            if np.max(arr) != np.min(arr):
                n = 0.0
            else:
                n = 1e-08
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + n)
        elif self.norm_mode == "z_score":
            std = np.std(arr)
            if std == 0:
                std += 1e-08
            arr = (arr - np.mean(arr)) / std

        arr = arr * self.scale
        return arr
