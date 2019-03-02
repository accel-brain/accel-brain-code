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
        self.__token_list = list(set(token_list))

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens..
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        vector_list = [self.__t_hot(token).tolist() for token in token_list]
        return vector_list

    def __t_hot(self, token):
        arr = np.zeros(len(self.__token_list))
        key = self.__token_list.index(token)
        arr[key] = 1
        arr = arr.astype(np.float32)
        return arr
