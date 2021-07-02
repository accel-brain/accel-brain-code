# -*- coding: utf-8 -*-
import nltk
from pysummarization.vectorizable_token import VectorizableToken


class TfidfVectorizer(VectorizableToken):
    '''
    Vectorize token.
    '''
    # Document
    __collection = []

    __vector_list = None
    
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
        self.__vector_list = vector_list
        return vector_list

    def get_dim(self):
        ''' getter '''
        if self.__vector_list is None:
            _ = self.vectorize(["dummy"])
        
        return len(self.__vector_list)

    def set_dim(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    dim = property(get_dim, set_dim)

