# -*- coding: utf-8 -*-
from pysummarization.vectorizable_token import VectorizableToken
from gensim.models import Word2Vec


class Word2VecVectorizer(VectorizableToken):
    '''
    Vectorize token by word2vec.
    '''
    
    __word2vec_model = None

    def get_word2vec_model(self):
        ''' getter '''
        return self.__word2vec_model

    def set_word2vec_model(self, value):
        ''' setter '''
        if isinstance(value, Word2Vec) is True:
            self.__word2vec_model = value
        else:
            raise TypeError("The type of `word2vec_model` must be `gensim.models.Word2Vec`.")

    word2vec_model = property(get_word2vec_model, set_word2vec_model)

    def __init__(self, word2vec_model):
        '''
        Initialize.
        
        Args:
            token_list_list:    The list of list of tokens.
        '''
        if isinstance(word2vec_model, Word2Vec) is True:
            self.__word2vec_model = word2vec_model
        else:
            raise TypeError("The type of `word2vec_model` must be `gensim.models.Word2Vec`.")

    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens..
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        vector_list = [self.__word2vec_model.wv[token] for token in token_list]
        return vector_list

    __dim = None

    def get_dim(self):
        ''' getter ''' 
        return self.__dim

    def set_dim(self, value):
        ''' setter '''
        self.__dim = value
    
    dim = property(get_dim, set_dim)
