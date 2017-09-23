# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class TokenizableDoc(metaclass=ABCMeta):
    '''
    Tokenize string.
    
    '''
    
    @abstractmethod
    def tokenize(self, sentence_str):
        '''
        Tokenize str.
        
        Args:
            sentence_str:   tokenized string.
        
        Returns:
            [token, token, token, ...]
        '''
        raise NotImplementedError("This method must be implemented.")
