# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class VectorlizableToken(metaclass=ABCMeta):
    '''
    Vectorlize token.
    '''
    
    @abstractmethod
    def vectorlize(self, token_list):
        '''
        Tokenize token list.
        
        Args:
            token_list:   The list of tokens.
        
        Returns:
            [vector of token, vector of token, vector of token, ...]
        '''
        raise NotImplementedError("This method must be implemented.")
