# -*- coding: utf-8 -*-
from accelbrainbase.iteratable_data import IteratableData
from pysummarization.vectorizable_token import VectorizableToken
import numpy as np


class Word2VecIterator(IteratableData):
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
        sentence_list, 
        test_sentence_list=None,
        parts_of_speech_list=[],
        test_parts_of_speech_list=[],
        epochs=1000,
        batch_size=25,
        seq_len=2,
        noiseable_data=None,
        generation_flag=True,
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
        self.vectorizable_token = vectorizable_token
        self.sentence_list = sentence_list
        if test_sentence_list is None:
            self.test_sentence_list = sentence_list
        else:
            self.test_sentence_list = test_sentence_list

        all_token_list = sum(sentence_list, [])
        bigram_tuple_list = []
        for i in range(1, len(all_token_list)):
            bigram_tuple_list.append(
                (all_token_list[i-1], all_token_list[i])
            )
        self.bigram_tuple_list = bigram_tuple_list

        self.parts_of_speech_list = parts_of_speech_list
        self.test_parts_of_speech_list = test_parts_of_speech_list

        pos_master_list = list(set(sum(parts_of_speech_list, [])))
        self.pos_master_list = sorted(pos_master_list)

        token_pos_dict = {}
        for i in range(len(sentence_list)):
            for j in range(len(sentence_list[i])):
                token_pos_dict.setdefault(
                    (sentence_list[i][j], parts_of_speech_list[i][j]),
                    0
                )
                token_pos_dict[(sentence_list[i][j], parts_of_speech_list[i][j])] += 1

        if test_sentence_list is not None:
            for i in range(len(test_sentence_list)):
                for j in range(len(test_sentence_list[i])):
                    token_pos_dict.setdefault(
                        (test_sentence_list[i][j], test_parts_of_speech_list[i][j]),
                        0
                    )
                    token_pos_dict[(sentence_list[i][j], parts_of_speech_list[i][j])] += 1

        self.token_pos_dict = token_pos_dict

        dataset_size = len(sum(sentence_list, []))
        iter_n = int(epochs * max(dataset_size / batch_size, 1))

        self.iter_n = iter_n
        self.epochs = epochs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.__noiseable_data = noiseable_data
        self.generation_flag = generation_flag

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
            training_observed_list = []
            training_objected_list = []
            test_observed_list = []
            test_objected_list = []
            if len(self.parts_of_speech_list) > 0:
                training_pos_list = []
                test_pos_list = []

            for batch in range(self.batch_size):
                key = np.random.randint(low=0, high=len(self.sentence_list))
                token_list = self.sentence_list[key]
                if len(token_list) < self.seq_len:
                    raise ValueError("The length of sentence must be more than `seq_len`.")
                start_key = np.random.randint(low=0, high=len(token_list)-self.seq_len-1)
                vector_list = self.vectorizable_token.vectorize(
                    token_list[start_key:start_key+self.seq_len]
                )
                training_observed_list.append(vector_list)
                if self.generation_flag is True:
                    vector_list = self.vectorizable_token.vectorize(
                        [token_list[start_key+self.seq_len+1]]
                    )
                training_objected_list.append(vector_list)
                if len(self.parts_of_speech_list) > 0:
                    observed_pos_list = []
                    for pos in self.parts_of_speech_list[key][start_key:start_key+self.seq_len]:
                        pos_key = self.pos_master_list.index(pos)
                        arr = np.zeros(len(self.pos_master_list))
                        arr[pos_key] = 1
                        observed_pos_list.append(arr.tolist())

                    pos = self.parts_of_speech_list[key][start_key+self.seq_len+1]
                    pos_key = self.pos_master_list.index(pos)
                    arr = np.zeros(len(self.pos_master_list))
                    arr[pos_key] = 1
                    training_pos_list.append(arr.tolist())

                key = np.random.randint(low=0, high=len(self.test_sentence_list))
                token_list = self.test_sentence_list[key]

                if len(token_list) < self.seq_len:
                    raise ValueError("The length of sentence must be more than `seq_len`.")
                start_key = np.random.randint(low=0, high=len(token_list)-self.seq_len-1)
                vector_list = self.vectorizable_token.vectorize(
                    token_list[start_key:start_key+self.seq_len]
                )
                test_observed_list.append(vector_list)
                if self.generation_flag is True:
                    vector_list = self.vectorizable_token.vectorize(
                        [token_list[start_key+self.seq_len+1]]
                    )
                test_objected_list.append(vector_list)
                if len(self.test_parts_of_speech_list) > 0:
                    pos = self.test_parts_of_speech_list[key][start_key+self.seq_len+1]
                    pos_key = self.pos_master_list.index(pos)
                    arr = np.zeros(len(self.pos_master_list))
                    arr[pos_key] = 1
                    test_pos_list.append(arr.tolist())

            training_observed_arr = np.array(training_observed_list)
            training_objected_arr = np.array(training_objected_list)
            test_observed_arr = np.array(test_observed_list)
            test_objected_arr = np.array(test_objected_list)
            if len(self.parts_of_speech_list) > 0:
                training_pos_arr = np.array(training_pos_list)
                test_pos_arr = np.array(test_pos_list)

            if self.__noiseable_data is not None:
                training_observed_arr = self.__noiseable_data.noise(training_observed_arr)
                test_observed_arr = self.__noiseable_data.noise(test_observed_arr)

            if len(self.parts_of_speech_list) > 0:
                yield training_observed_arr, training_objected_arr, test_observed_arr, test_objected_arr, training_pos_arr, test_pos_arr
            else:
                yield training_observed_arr, training_objected_arr, test_observed_arr, test_objected_arr

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
        test_observed_list = []
        for key in range(self.test_sentence_list):
            token_list = self.test_sentence_list[key]

            if len(token_list) < self.seq_len:
                raise ValueError("The length of sentence must be more than `seq_len`.")

            for start_key in range(len(token_list)-self.seq_len):
                vector_list = self.vectorizable_token.vectorize(
                    token_list[start_key:start_key+self.seq_len]
                )
                test_observed_list.append(vector_list)
                if len(test_observed_list) == self.batch_size:
                    test_observed_arr = np.array(test_observed_list)
                    yield None, None, test_observed_arr, None


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
