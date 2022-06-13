# -*- coding: utf-8 -*-
from pysummarization.iteratabledata.word2vec_iterator import Word2VecIterator as _Word2VecIterator
import torch
import numpy as np


class Word2VecIterator(_Word2VecIterator):
    '''
    Word2Vec Iterator.
    '''

    def __init__(
        self, 
        vectorizable_token, 
        sentence_list, 
        test_sentence_list=None,
        parts_of_speech_list=[],
        test_parts_of_speech_list=[],
        epochs=1000,
        batch_size=25,
        seq_len=2,
        noiseable_data=None,
        generation_flag=True,
        ctx="cpu"
    ):
        '''
        Init.

        Args:
            vectorizable_token:     is-a `VectorizableToken`.
            sentence_list:          `list` of `list`s of tokens.
            parts_of_speech_list:   `list` of parts of speech.
            epochs:                 `int` of epochs.
            batch_size:             `int` of batch size.
            seq_len:                `int` of length of series.
            noiseable_data:         is-a `NoiseableData`.
            generation_flag:        `bool`.
                                    If `True`, this iterator outputs observed data points and supervised data as [t_1, t_2, t_3] and [t_2, t_3, t_4].
                                    If `False`, this iterator outputs observed data points and supervised data as [t_1, t_2, t_3] and [t_1, t_2, t_3].
        '''
        super().__init__(
            vectorizable_token=vectorizable_token, 
            sentence_list=sentence_list, 
            test_sentence_list=test_sentence_list,
            parts_of_speech_list=parts_of_speech_list,
            test_parts_of_speech_list=test_parts_of_speech_list,
            epochs=epochs,
            batch_size=batch_size,
            seq_len=seq_len,
            generation_flag=generation_flag,
            noiseable_data=None
        )
        self.__ctx = ctx
        self.__noiseable_data_ = noiseable_data

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
        for arr_tuple in super().generate_learned_samples():
            if len(self.parts_of_speech_list) > 0:
                training_observed_arr, training_objected_arr, test_observed_arr, test_objected_arr, training_pos_arr, test_pos_arr = arr_tuple
            else:
                training_observed_arr, training_objected_arr, test_observed_arr, test_objected_arr = arr_tuple

            training_observed_arr = torch.from_numpy(training_observed_arr)
            training_observed_arr = training_observed_arr.to(self.__ctx)
            training_objected_arr = torch.from_numpy(training_objected_arr)
            training_objected_arr = training_objected_arr.to(self.__ctx)
            test_observed_arr = torch.from_numpy(test_observed_arr)
            test_observed_arr = test_observed_arr.to(self.__ctx)
            test_objected_arr = torch.from_numpy(test_objected_arr)
            test_objected_arr = test_objected_arr.to(self.__ctx)

            if self.__noiseable_data_ is not None:
                training_observed_arr = self.__noiseable_data_.noise(training_observed_arr)
                test_observed_arr = self.__noiseable_data_.noise(test_observed_arr)

            if len(self.parts_of_speech_list) > 0:
                training_pos_arr = torch.from_numpy(training_pos_arr)
                training_pos_arr = training_pos_arr.to(self.__ctx)
                test_pos_arr = torch.from_numpy(test_pos_arr)
                test_pos_arr = test_pos_arr.to(self.__ctx)

            if len(self.parts_of_speech_list) > 0:
                yield training_observed_arr.float(), training_objected_arr.float(), test_observed_arr.float(), test_objected_arr.float(), training_pos_arr.float(), test_pos_arr.float()
            else:
                yield training_observed_arr.float(), training_objected_arr.float(), test_observed_arr.float(), test_objected_arr.float()

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
