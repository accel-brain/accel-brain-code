# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.vectorizable_token import VectorizableToken


class THotVectorizer(VectorizableToken):
    '''
    Vectorize token by t-hot Vectorizer.
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
        return [self.__t_hot(token).tolist() for token in token_list]

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

    def get_token_arr(self):
        ''' getter '''
        return self.__token_arr
    
    def set_token_arr(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    token_arr = property(get_token_arr, set_token_arr)
