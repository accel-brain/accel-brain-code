# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.nlp_base import NlpBase
from pysummarization.tokenizable_doc import TokenizableDoc
from pysummarization.vectorizable_token import VectorizableToken
from pysummarization.computable_similarity import ComputableSimilarity


class IterTextGenerator(object):
    '''
    Iterator/Generator that generates vectors of sentence.
    '''

    def __init__(
        self,
        document,
        nlp_base,
        tokenizable_doc,
        vectorizable_token,
        computable_similarity,
        epochs=10,
        batch_size=20,
        seq_len=10
    ):
        '''
        Init.

        Args:
            document:               `str` of all sentence.
            tokenizable_doc:        is-a `TokenizableDoc`.
            vectorizable_token:     is-a `VectorizableToken`.
            computable_similarity:  is-a `ComputableSimilarity`.
            epochs:                 Epochs.
            batch_size:             Batch size.
            seq_len:                The length of sequence.
        '''
        if isinstance(tokenizable_doc, TokenizableDoc) is False:
            raise TypeError()
        if isinstance(vectorizable_token, VectorizableToken) is False:
            raise TypeError()
        if isinstance(computable_similarity, ComputableSimilarity) is False:
            raise TypeError()

        token_list = tokenizable_doc.tokenize(document)
        vector_list = []
        max_len = 0
        for i in range(len(token_list)):
            vec_list = vectorizable_token.vectorize(
                token_list[i]
            )
            vector_list.append(vec_list)
            if max_len < len(vec_list):
                max_len = len(vec_list)
        
        for i in range(len(vector_list)):
            while max_len > len(vector_list[i]):
                vector_list[i].append(0.0)

        self.__token_arr = np.array(token_list)
        self.__vector_arr = np.array(vector_list)

        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__seq_len = seq_len

        self.__computable_similarity = computable_similarity

    def generate_uniform(self):
        '''
        Draw the samples from uniform distribution.

        Returns:
            Tuple data.
            - `np.ndarray` of vectors.
            - `np.ndarray` of tokens.
        '''
        index_arr = np.arange(self.__token_arr.shape[0])
        np.random.shuffle(index_arr)
        for i in range(self.__epochs):
            vector_arr = np.empty((self.__batch_size, self.__seq_len, self.vector_dim))
            token_arr = np.empty((self.__batch_size, self.__seq_len), dtype=object)
            for row in range(self.__batch_size):
                for seq_len in range(self.__seq_len):
                    key = np.random.randint(
                        low=0,
                        high=self.__token_arr.shape[0] - self.__seq_len, 
                        size=1
                    )[0]
                    vector_arr[row, :] = self.__vector_arr[index_arr][key:key+self.__seq_len]
                    token_arr[row, :] = self.__token_arr[index_arr][key:key+self.__seq_len]

            yield vector_arr, token_arr

    def generate_real_token(self):
        '''
        Draw the samples from real sentences.

        Returns:
            Tuple data.
            - `np.ndarray` of vectors.
            - `np.ndarray` of tokens.
        '''
        for i in range(self.__epochs):
            vector_arr = np.empty((self.__batch_size, self.__seq_len, self.vector_dim))
            token_arr = np.empty((self.__batch_size, self.__seq_len), dtype=object)
            for row in range(self.__batch_size):
                for seq_len in range(self.__seq_len):
                    key = np.random.randint(
                        low=0,
                        high=self.__token_arr.shape[0] - self.__seq_len, 
                        size=1
                    )[0]
                    vector_arr[row, :] = self.__vector_arr[key:key+self.__seq_len]
                    token_arr[row, :] = self.__token_arr[key:key+self.__seq_len]

            yield vector_arr, token_arr

    def convert_vector_into_token(self, vector_arr):
        '''
        Convert vectors into token, which is similar to the input vectors.

        Args:
            vector_arr:     `np.ndarray` of vector.
        
        Returns:
            Tuple data.
            - `np.ndarray` of vectors.
            - `np.ndarray` of tokens.
            - `list` of sentences.
        '''
        similarity_arr = self.__computable_similarity.compute(vector_arr, self.__sentence_vector_arr)
        return (
            self.__sentence_vector_arr[np.argmax(similarity_arr)],
            self.__sentence_token_arr[np.argmax(similarity_arr)],
            self.__sentence_list[np.argmax(similarity_arr)]
        )

    def get_vector_dim(self):
        ''' getter '''
        return self.__vector_arr.shape[-1]
    
    def set_vector_dim(self, value):
        ''' setter '''
        raise TypeError("The `vector_dim` must be read-only.")
    
    vector_dim = property(get_vector_dim, set_vector_dim)
