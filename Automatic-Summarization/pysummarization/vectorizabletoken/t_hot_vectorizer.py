# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.vectorizable_token import VectorizableToken


class THotVectorizer(VectorizableToken):
    '''
    Vectorize token.
    '''    

    def __init__(self, token_list):
        '''
        Initialize.
        
        Args:
            token_list:    The list of all tokens.
        '''
        self.__token_arr = np.array(list(set(token_list)))

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        vector_list = [self.__t_hot(token).tolist() for token in token_list]
        return vector_list

    def tokenize(self, vector_list):
        '''
        Tokenize vector.

        Args:
            vector_list:    The list of vector of one token.
        
        Returns:
            token
        '''
        vector_arr = np.array(vector_list)
        if vector_arr.ndim == 1:
            key_arr = vector_arr.argmax()
        else:
            key_arr = vector_arr.argmax(axis=-1)
        return self.__token_arr[key_arr]

    def __t_hot(self, token):
        arr = np.zeros(len(self.__token_arr))
        key = self.__token_arr.tolist().index(token)
        arr[key] = 1
        arr = arr.astype(np.float32)
        return arr
