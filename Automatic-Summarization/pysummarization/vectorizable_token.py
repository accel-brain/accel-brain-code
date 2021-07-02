# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class VectorizableToken(metaclass=ABCMeta):
    '''
    Vectorize token.
    '''
    
    @abstractmethod
    def vectorize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractproperty
    def dim(self):
        ''' `int` of dimension of vectors. '''
        raise NotImplementedError()
