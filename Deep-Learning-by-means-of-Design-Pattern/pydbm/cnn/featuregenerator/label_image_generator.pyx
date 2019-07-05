# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.cnn.feature_generator import FeatureGenerator
from PIL import Image
import os
import numpy as np
cimport numpy as np
ctypedef np.float64_t DOUBLE_t


class LabelImageGenerator(FeatureGenerator):
    '''
    Image generator for Auto-Encoder or Encoder/Decoder scheme.
    '''
    # Epochs of Mini-batch.
    __epochs = 1000
    # Batch size of Mini-batch.
    __batch_size = 20

    def __init__(
        self,
        epochs,
        batch_size,
        training_image_dir,
        test_image_dir,
        seq_len=None,
        gray_scale_flag=True,
        wh_size_tuple=None,
        norm_mode="z_score",
    ):
        '''
        Init.

        Args:
            epochs:                         Epochs of Mini-batch.
            bath_size:                      Batch size of Mini-batch.
            training_image_dir:             Dir path which stores image files for training.
            test_image_dir:                 Dir path which stores image files for test.
            seq_len:                        The length of one sequence.
            gray_scale_flag:                Gray scale or not(RGB).
            wh_size_tuple:                  Tuple(`width`, `height`).
            norm_mode:                      How to normalize pixel values of images.
                                            - `z_score`: Z-Score normalization.
                                            - `min_max`: Min-max normalization.
                                            - `tanh`: Normalization by tanh function.
        '''
        self.__epochs = epochs
        self.__batch_size = batch_size

        dir_name_list = os.listdir(training_image_dir)
        self.__training_file_path_list_dict = {}
        for dir_name in dir_name_list:
            file_name_list = os.listdir(training_image_dir + "/" + dir_name + "/")
            self.__training_file_path_list_dict.setdefault(
                dir_name,
                [training_image_dir + "/" + dir_name + "/" + file_name for file_name in file_name_list]
            )
        self.__label_list = dir_name_list
        if seq_len is not None and len(self.__training_file_path_list) < seq_len:
            raise ValueError("The `seq_len` must be more than the number of files which are stored in `training_image_dir`.")

        dir_name_list = os.listdir(test_image_dir)
        self.__test_file_path_list_dict = {}
        for dir_name in dir_name_list:
            file_name_list = os.listdir(test_image_dir + "/" + dir_name + "/")
            self.__test_file_path_list_dict.setdefault(
                dir_name,
                [test_image_dir + "/" + dir_name + "/" + file_name for file_name in file_name_list]
            )
        if seq_len is not None and len(self.__test_file_path_list) < seq_len:
            raise ValueError("The `seq_len` must be more than the number of files which are stored in `test_image_dir`.")

        self.__seq_len = seq_len
        self.__gray_scale_flag = gray_scale_flag
        self.__wh_size_tuple = wh_size_tuple
        self.__norm_mode = norm_mode

        logger = getLogger("pydbm")
        self.__logger = logger

    def generate(self):
        '''
        Generate feature points.
        
        Returns:
            The tuple of feature points.
            The shape is: (`Training data`, `Training label`, `Test data`, `Test label`).
        '''
        cdef np.ndarray _training_data_arr
        cdef np.ndarray _training_label_arr
        cdef np.ndarray _test_data_arr
        cdef np.ndarray _test_label_arr

        for epoch in range(self.epochs):
            _training_data_arr, _training_label_arr, _test_data_arr, _test_label_arr = None, None, None, None
            for batch_size in range(self.batch_size):
                if self.__seq_len is None:
                    dir_key = self.__label_list[
                        np.random.randint(low=0, high=len(self.__label_list))
                    ]
                    file_key = np.random.randint(
                        low=0, 
                        high=len(self.__training_file_path_list_dict[dir_key])
                    )
                    training_data_arr = self.__read(self.__training_file_path_list_dict[dir_key][file_key])
                    training_label_arr = np.zeros((1, len(self.__label_list)))
                    training_label_arr[0, self.__label_list.index(dir_key)] = 1

                    file_key = np.random.randint(
                        low=0, 
                        high=len(self.__test_file_path_list_dict[dir_key])
                    )
                    test_data_arr = self.__read(self.__test_file_path_list_dict[dir_key][file_key])
                    test_label_arr = training_label_arr.copy()
                else:
                    training_data_arr, test_data_arr = None, None
                    dir_key = self.__label_list[
                        np.random.randint(low=0, high=len(self.__label_list))
                    ]
                    training_file_key = np.random.randint(
                        low=0,
                        high=len(self.__training_file_path_list_dict[dir_key]) - self.__seq_len
                    )
                    test_file_key = np.random.randint(
                        low=0,
                        high=len(self.__test_file_path_list_dict[dir_key]) - self.__seq_len
                    )
                    for seq in range(self.__seq_len):
                        seq_training_data_arr = self.__read(
                            self.__training_file_path_list_dict[dir_key][training_file_key+seq]
                        )
                        seq_test_data_arr = self.__read(
                            self.__test_file_path_list_dict[dir_key][test_file_key+seq]
                        )
                        
                        if training_data_arr is not None:
                            training_data_arr = np.r_[training_data_arr, seq_training_data_arr]
                        else:
                            training_data_arr = seq_training_data_arr
                        
                        if test_data_arr is not None:
                            test_data_arr = np.r_[test_data_arr, seq_test_data_arr]
                        else:
                            test_data_arr = seq_test_data_arr

                    training_data_arr = np.expand_dims(training_data_arr, axis=0)
                    test_data_arr = np.expand_dims(test_data_arr, axis=0)
                    training_label_arr = np.zeros((1, len(self.__label_list)))
                    training_label_arr[0, self.__label_list.index(dir_key)] = 1
                    test_label_arr = training_label_arr.copy()

                if _training_data_arr is not None:
                    _training_data_arr = np.r_[_training_data_arr, training_data_arr]
                else:
                    _training_data_arr = training_data_arr
                
                if _training_label_arr is not None:
                    _training_label_arr = np.r_[_training_label_arr, training_label_arr]
                else:
                    _training_label_arr = training_label_arr
                if _test_data_arr is not None:
                    _test_data_arr = np.r_[_test_data_arr, test_data_arr]
                else:
                    _test_data_arr = test_data_arr

                if _test_label_arr is not None:
                    _test_label_arr = np.r_[_test_label_arr, test_label_arr]
                else:
                    _test_label_arr = test_label_arr

            yield _training_data_arr, _training_label_arr, _test_data_arr, _test_label_arr

    def __read(self, file_path):
        '''
        Read image file.
        
        Args:
            file_path:  An image file path.
        
        Returns:
            `np.ndarray` of image file.
        '''
        img = Image.open(file_path)
        if self.__wh_size_tuple is not None:
            img = img.resize(self.__wh_size_tuple)
        if self.__gray_scale_flag is True:
            img = img.convert("L")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:
            arr = arr.transpose((2, 0, 1))
            arr = np.expand_dims(arr, axis=0)

        if self.__norm_mode is not None:
            if self.__norm_mode == "z_score":
                arr = (arr - arr.mean()) / arr.std()
            elif self.__norm_mode == "min_max":
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            elif self.__norm_mode == "tanh":
                arr = arr.astype(np.float64)
                arr = arr - arr.mean()
                arr = np.tanh(arr)

        return arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_epochs(self):
        ''' getter '''
        return self.__epochs

    def get_batch_size(self):
        ''' getter '''
        return self.__batch_size

    epochs = property(get_epochs, set_readonly)
    batch_size = property(get_batch_size, set_readonly)
