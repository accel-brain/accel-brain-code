# -*- coding: utf-8 -*-
import nltk
from pysummarization.vectorizable_token import VectorizableToken


class TfidfVectorizer(VectorizableToken):
    '''
    Vectorize token.
    '''
    # Document
    __collection = []
    
    def __init__(self, token_list_list):
        '''
        Initialize.
        
        Args:
            token_list_list:    The list of list of tokens.
        '''
        self.__collection = nltk.TextCollection(token_list_list)

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens..
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        vector_list = [self.__collection.tf_idf(token, self.__collection) for token in token_list]
        return vector_list
