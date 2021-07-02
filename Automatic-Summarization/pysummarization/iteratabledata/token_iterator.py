# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from pysummarization.vectorizable_token import VectorizableToken
import numpy as np


class TokenIterator(IteratableData):
    '''
    Token Iterator.
    '''

    # is-a `VectorizableToken`.
    __vectorizable_token = None

    def get_vectorizable_token(self):
        ''' getter '''
        return self.__vectorizable_token
    
    def set_vectorizable_token(self, value):
        ''' setter '''
        if isinstance(value, VectorizableToken) is False:
            raise TypeError("The type of `vectorizable_token` must be `VectorizableToken`.")
        self.__vectorizable_token = value
    
    vectorizable_token = property(get_vectorizable_token, set_vectorizable_token)

    def __init__(
        self, 
        vectorizable_token, 
        token_arr, 
        epochs=1000,
        batch_size=25,
        seq_len=5,
        test_size=0.3,
        norm_mode=None,
        noiseable_data=None
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
        self.vectorizable_token = vectorizable_token
        vector_list = vectorizable_token.vectorize(token_list=token_arr.tolist())
        vector_arr = np.array(vector_list)

        observed_list = []
        for i in range(seq_len, vector_arr.shape[0]):
            observed_list.append(vector_arr[i-seq_len:i])
        observed_arr = np.array(observed_list)

        print("setup observed arr: " + str(observed_arr.shape))

        self.observed_arr = observed_arr

        training_row = int(observed_arr.shape[0] * (1 - test_size))
        key_arr = np.arange(observed_arr.shape[0])
        np.random.shuffle(key_arr)
        training_arr = observed_arr[key_arr[:training_row]]
        test_arr = observed_arr[key_arr[training_row:]]

        dataset_size = training_arr.shape[0]
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.training_arr = training_arr
        self.test_arr = test_arr

        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.__norm_mode = norm_mode
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
            training_key_arr = np.arange(self.training_arr.shape[0])
            test_key_arr = np.arange(self.test_arr.shape[0])

            np.random.shuffle(training_key_arr)
            np.random.shuffle(test_key_arr)

            training_batch_arr = self.training_arr[training_key_arr[:self.batch_size]]
            test_batch_arr = self.test_arr[test_key_arr[:self.batch_size]]

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
        i = 0
        while i + self.batch_size < self.observed_arr.shape[0]:
            test_batch_arr = self.observed_arr[i:i+self.batch_size]
            test_batch_arr = self.pre_normalize(test_batch_arr)
            i = i + self.batch_size

            yield None, None, test_batch_arr, None

    def pre_normalize(self, arr):
        '''
        Normalize before observation.

        Args:
            arr:    Tensor.
        
        Returns:
            Tensor.
        '''
        if self.__norm_mode == "min_max":
            if arr.max() != arr.min():
                n = 0.0
            else:
                n = 1e-08
            arr = (arr - arr.min()) / (arr.max() - arr.min() + n)
        elif self.__norm_mode == "z_score":
            std = arr.asnumpy().std()
            if std == 0:
                std += 1e-08
            arr = (arr - arr.mean()) / std

        arr = arr * self.__scale
        return arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_epochs(self):
        ''' getter '''
        return self.__epochs

    def set_epochs(self, value):
        ''' setter '''
        self.__epochs = value

    epochs = property(get_epochs, set_epochs)

    def get_batch_size(self):
        ''' getter '''
        return self.__batch_size

    def set_batch_size(self, value):
        ''' setter '''
        self.__batch_size = value

    batch_size = property(get_batch_size, set_batch_size)

    def get_seq_len(self):
        ''' getter '''
        return self.__seq_len
    
    def set_seq_len(self, value):
        ''' setter '''
        self.__seq_len = value

    seq_len = property(get_seq_len, set_seq_len)

    __norm_mode = "z_score"

    def get_norm_mode(self):
        ''' getter '''
        return self.__norm_mode
    
    def set_norm_mode(self, value):
        ''' setter '''
        self.__norm_mode = value
    
    norm_mode = property(get_norm_mode, set_norm_mode)

    __scale = 1.0

    def get_scale(self):
        ''' getter '''
        return self.__scale
    
    def set_scale(self, value):
        ''' setter '''
        self.__scale = value
    
    scale = property(get_scale, set_scale)
