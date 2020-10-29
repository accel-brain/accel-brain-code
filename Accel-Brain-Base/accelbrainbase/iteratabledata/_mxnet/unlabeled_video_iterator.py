# -*- coding: utf-8 -*-
import numpy as np
import mxnet.ndarray as nd
from logging import getLogger
import os
import random

from accelbrainbase.extractabledata.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata.unlabeled_image_iterator import UnlabeledImageIterator as _UnlabeledImageIterator
from accelbrainbase.noiseable_data import NoiseableData


class UnlabeledVideoIterator(_UnlabeledImageIterator):
    '''
    Iterator that draws from image files and generates `mxnet.ndarray` of unlabeled samples.
    '''

    def __init__(
        self, 
        image_extractor, 
        dir_list, 
        test_dir_list=None,
        epochs=300,
        batch_size=20,
        seq_len=10,
        at_intervals=1,
        norm_mode="z_score",
        scale=1.0,
        noiseable_data=None,
    ):
        '''
        Init.

        Args:
            image_extractor:                is-a `ImageExtractor`.
            dir_list:                       `list` of directories that store your image files in training.
                                            This class will not scan the directories recursively and consider that
                                            all image file will be sorted by any rule in relation to your sequencial modeling.

            test_dir_list:                  `list` of directories that store your image files in test.
                                            If `None`, this value will be equivalent to `dir_list`.
                                            This class will not scan the directories recursively and consider that
                                            all image file will be sorted by any rule in relation to your sequencial modeling.

            epochs:                         `int` of epochs of Mini-batch.
            batch_size:                     `int` of batch size of Mini-batch.
            seq_len:                        `int` of the length of series.
            at_intervals:                   `int` of intervals in one sequence.

            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - others : This class will not normalize the data.

            scale:                          `float` of scaling factor for data.
            noiseable_data:                 is-a `NoiseableData` for Denoising Auto-Encoders.
        '''

        if isinstance(image_extractor, ImageExtractor) is False:
            raise TypeError("The type of `image_extractor` must be `ImageExtractor`.")
        if isinstance(dir_list, list) is False:
            raise TypeError("The type of `dir_list` must be `list`.")
        if noiseable_data is not None and isinstance(noiseable_data, NoiseableData) is False:
            raise TypeError("The type of `noiseable_data` must be `NoiseableData`.")
        if isinstance(seq_len, int) is False:
            raise TypeError("The type of `seq_len` must be `int`.")
        if seq_len <= 0:
            raise ValueError("The value of `seq_len` must be more than `0`.")
        if isinstance(at_intervals, int) is False:
            raise TypeError("The type of `at_intervals` must be `int`.")
        if at_intervals <= 0:
            raise ValueError("The value of `at_intervals` must be more than `0`.")

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        dir_list.sort()

        self.__training_file_path_list = [None] * len(dir_list)
        dataset_size = 0
        for i in range(len(dir_list)):
            file_path_list = [dir_list[i] + "/" + file_name for file_name in os.listdir(dir_list[i] + "/")]
            file_path_list.sort()
            self.__training_file_path_list[i] = file_path_list
            dataset_size = dataset_size + len(file_path_list)

        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        if test_dir_list is not None and isinstance(test_dir_list, list) is False:
            raise TypeError("The type of `test_dir_list` must be `list`.")
        elif test_dir_list is None:
            test_dir_list = dir_list
            self.__test_file_path_list = self.__training_file_path_list
        else:
            test_dir_list.sort()
            self.__test_file_path_list = [None] * len(test_dir_list)
            for i in range(len(test_dir_list)):
                file_path_list = [test_dir_list[i] + "/" + file_name for file_name in os.listdir(test_dir_list[i] + "/")]
                file_path_list.sort()
                self.__test_file_path_list[i] = file_path_list

        self.__image_extractor = image_extractor
        self.__dir_list = dir_list
        self.__test_dir_list = test_dir_list
        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.__seq_len = seq_len
        self.__at_intervals = at_intervals

        self.norm_mode = norm_mode

        self.scale = scale
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
        for _ in range(self.iter_n):
            training_batch_arr, test_batch_arr = None, None
            for batch_size in range(self.batch_size):
                dir_key = np.random.randint(low=0, high=len(self.__training_file_path_list))
                training_file_path_list = self.__split_at_intervals(
                    self.__training_file_path_list[dir_key], 
                    start_pos=0, 
                    seq_interval=self.__at_intervals
                )

                training_data_arr, test_data_arr = None, None
                training_file_key = np.random.randint(
                    low=0,
                    high=len(training_file_path_list) - self.__seq_len
                )

                test_dir_key = np.random.randint(low=0, high=len(self.__test_file_path_list))
                test_file_path_list = self.__split_at_intervals(
                    self.__test_file_path_list[test_dir_key], 
                    start_pos=0, 
                    seq_interval=self.__at_intervals
                )

                test_file_key = np.random.randint(
                    low=0,
                    high=len(test_file_path_list) - self.__seq_len
                )
                for seq in range(self.__seq_len):
                    seq_training_batch_arr = self.__image_extractor.extract(
                        path=training_file_path_list[training_file_key+seq],
                    )
                    seq_training_batch_arr = self.pre_normalize(seq_training_batch_arr)
                    seq_training_batch_arr = nd.expand_dims(seq_training_batch_arr, axis=0)
                    seq_test_batch_arr = self.__image_extractor.extract(
                        path=test_file_path_list[test_file_key+seq],
                    )
                    seq_test_batch_arr = self.pre_normalize(seq_test_batch_arr)
                    seq_test_batch_arr = nd.expand_dims(seq_test_batch_arr, axis=0)

                    if training_data_arr is not None:
                        training_data_arr = nd.concat(training_data_arr, seq_training_batch_arr, dim=0)
                    else:
                        training_data_arr = seq_training_batch_arr
                    
                    if test_data_arr is not None:
                        test_data_arr = nd.concat(test_data_arr, seq_test_batch_arr, dim=0)
                    else:
                        test_data_arr = seq_test_batch_arr

                training_data_arr = nd.expand_dims(training_data_arr, axis=0)
                test_data_arr = nd.expand_dims(test_data_arr, axis=0)

                if training_batch_arr is not None:
                    training_batch_arr = nd.concat(training_batch_arr, training_data_arr, dim=0)
                else:
                    training_batch_arr = training_data_arr

                if test_batch_arr is not None:
                    test_batch_arr = nd.concat(test_batch_arr, test_data_arr, dim=0)
                else:
                    test_batch_arr = test_data_arr

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
        scan_file_path_list = []
        for dir_key in range(len(self.__test_file_path_list)):
            for file_key in range(len(self.__test_file_path_list[dir_key]) - self.__seq_len):
                _scan_file_path_list = []
                for seq in range(self.__seq_len):
                    file_path = self.__test_file_path_list[dir_key][file_key+seq]
                    _scan_file_path_list.append(file_path)
                scan_file_path_list.append(_scan_file_path_list)

        random.shuffle(scan_file_path_list)

        test_batch_arr = None
        file_path_list = []
        for _file_path_list_ in scan_file_path_list:
            test_data_arr = None
            for seq in range(self.__seq_len):
                file_path = _file_path_list_[seq]
                seq_test_batch_arr = self.__image_extractor.extract(
                    path=file_path,
                )
                seq_test_batch_arr = self.pre_normalize(seq_test_batch_arr)
                seq_test_batch_arr = nd.expand_dims(seq_test_batch_arr, axis=0)
                if test_data_arr is not None:
                    test_data_arr = nd.concat(test_data_arr, seq_test_batch_arr, dim=0)
                else:
                    test_data_arr = seq_test_batch_arr

            test_data_arr = nd.expand_dims(test_data_arr, axis=0)
            if test_batch_arr is not None:
                test_batch_arr = nd.concat(test_batch_arr, test_data_arr, dim=0)
            else:
                test_batch_arr = test_data_arr

            file_path_list.append(file_path)

            if test_batch_arr.shape[0] == self.batch_size:
                test_batch_arr_ = test_batch_arr
                test_batch_arr = None
                _file_path_list = file_path_list
                file_path_list = []
                yield None, None, test_batch_arr_, _file_path_list
            elif test_batch_arr.shape[0] > self.batch_size:
                raise ValueError()

    def __split_at_intervals(self, file_path_list, start_pos=0, seq_interval=1):
        file_path_arr = np.array(file_path_list)
        return file_path_arr[list(range(start_pos, len(file_path_list), seq_interval))].tolist()
