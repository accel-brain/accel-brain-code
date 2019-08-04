# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.vectorizable_token import VectorizableToken
from pydbm.activation.tanh_function import TanhFunction
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
import numpy as np
import pandas as pd


class DBMLikeSkipGramVectorizer(VectorizableToken):
    '''
    Vectorize token by Deep Bolzmann Machine(DBM).

    Note that this class employs an original method 
    based on this library-specific intuition and analogy about skip-gram,
    where by n-grams are still stored to model language, 
    but they allow for tokens to be skipped.
    '''    

    def __init__(
        self, 
        token_list, 
        document_list=[],
        traning_count=100,
        batch_size=20,
        learning_rate=1e-05,
        feature_dim=100
    ):
        '''
        Initialize.
        
        Args:
            token_list:         The list of all tokens in all sentences.
                                If the input value is a two-dimensional list, 
                                the first-dimensional key represents a sentence number, 
                                and the second-dimensional key represents a token number.

            document_list:      The list of document composed by tokens.
            training_count:     The epochs.
            batch_size:         Batch size.
            learning_rate:      Learning rate.
            feature_dim:        The dimension of feature points.
        '''
        pair_dict = {}
        document_dict = {}

        self.__token_arr = np.array(token_list)
        if self.__token_arr.ndim == 2:
            for i in range(self.__token_arr.shape[0]):
                for j in range(1, self.__token_arr[i].shape[0] - 1):
                    pair_dict.setdefault((self.__token_arr[i, j], self.__token_arr[i, j-1]), 0)
                    pair_dict[(self.__token_arr[i, j], self.__token_arr[i, j-1])] += 1
                    pair_dict.setdefault((self.__token_arr[i, j], self.__token_arr[i, j+1]), 0)
                    pair_dict[(self.__token_arr[i, j], self.__token_arr[i, j+1])] += 1
                    document_dict.setdefault(self.__token_arr[i], [])
                    for d in range(len(document_list)):
                        if self.__token_arr[i, j] in document_list[d]:
                            document_dict[self.__token_arr[i, j]].append(d)

        elif self.__token_arr.ndim == 1:
            for i in range(1, self.__token_arr.shape[0] - 1):
                pair_dict.setdefault((self.__token_arr[i], self.__token_arr[i-1]), 0)
                pair_dict[(self.__token_arr[i], self.__token_arr[i-1])] += 1
                pair_dict.setdefault((self.__token_arr[i], self.__token_arr[i+1]), 0)
                pair_dict[(self.__token_arr[i], self.__token_arr[i+1])] += 1

                document_dict.setdefault(self.__token_arr[i], [])
                for d in range(len(document_list)):
                    if self.__token_arr[i] in document_list[d]:
                        document_dict[self.__token_arr[i]].append(d)
        else:
            raise ValueError()

        token_list = list(set(self.__token_arr.ravel().tolist()))

        token_arr = np.zeros((len(token_list), len(token_list)))
        pair_arr = np.zeros((len(token_list), len(token_list)))
        document_arr = np.zeros((len(token_list), len(document_list)))
        for i in range(token_arr.shape[0]):
            for j in range(token_arr.shape[0]):
                try:
                    pair_arr[i, j] = pair_dict[(token_list[i], token_list[j])]
                    token_arr[i, j] = 1.0
                except:
                    pass
            
            if len(document_list) > 0:
                if token_list[i] in document_dict:
                    for d in document_dict[token_list[i]]:
                        document_arr[i, d] = 1.0

        pair_arr = np.exp(pair_arr - pair_arr.max())
        pair_arr = pair_arr / pair_arr.sum()
        pair_arr = (pair_arr - pair_arr.mean()) / (pair_arr.std() + 1e-08)
        if len(document_list) > 0:
            document_arr = (document_arr - document_arr.mean()) / (document_arr.std() + 1e-08)

            token_arr = np.c_[pair_arr, document_arr]
            token_arr = (token_arr - token_arr.mean()) / (token_arr.std() + 1e-08)

        self.__dbm = StackedAutoEncoder(
            DBMMultiLayerBuilder(),
            [token_arr.shape[1], feature_dim, token_arr.shape[1]],
            [TanhFunction(), TanhFunction(), TanhFunction()],
            [ContrastiveDivergence(), ContrastiveDivergence()],
            learning_rate=learning_rate
        )
        self.__dbm.learn(token_arr, traning_count=traning_count, batch_size=batch_size, sgd_flag=True)
        feature_points_arr = self.__dbm.feature_points_arr
        self.__token_arr = token_arr
        self.__token_list = token_list
        self.__feature_points_arr = feature_points_arr

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        return [self.__extract_from_feature(token).tolist() for token in token_list]

    def convert_tokens_into_matrix(self, token_list):
        '''
        Create matrix of sentences.

        Args:
            token_list:     The list of tokens.
        
        Returns:
            2-D `np.ndarray` of sentences.
            Each row means one hot vectors of one sentence.
        '''
        return np.array(self.vectorize(token_list)).astype(np.float32)

    def tokenize(self, vector_list):
        '''
        Tokenize vector.

        Args:
            vector_list:    The list of vector of one token.
        
        Returns:
            token
        '''
        vector_arr = np.array(vector_list)
        if vector_arr.ndim == 2 and vector_arr.shape[0] > 1:
            vector_arr = np.nanmean(vector_arr, axis=0)

        vector_arr = vector_arr.reshape(1, -1)
        diff_arr = np.nansum(np.square(vector_arr - self.__feature_points_arr), axis=1)
        return np.array([self.__token_list[diff_arr.argmin(axis=0)]])

    def __extract_from_feature(self, token):
        try:
            key = self.__token_list.index(token)
            arr = self.__feature_points_arr[key]
            arr = arr.astype(np.float32)
        except:
            arr = self.__feature_points_arr.mean(axis=0)
        return arr

    def get_token_arr(self):
        ''' getter '''
        return self.__token_arr
    
    def set_token_arr(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    token_arr = property(get_token_arr, set_token_arr)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    def get_token_list(self):
        ''' getter '''
        return self.__token_list
    
    token_list = property(get_token_list, set_readonly)
