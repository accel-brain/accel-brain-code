# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class VectorlizableSentence(metaclass=ABCMeta):
    '''
    Vectorlize sentence.
    '''
    
    @abstractmethod
    def vectorlize(self, sentence_list):
        '''
        Tokenize token list.
        
        Args:
            sentence_list:   The list of tokenized sentences:
                             [
                                 [`token`, `token`, `token`, ...],
                                 [`token`, `token`, `token`, ...],
                                 [`token`, `token`, `token`, ...],
                             ]
        
        Returns:
            `np.ndarray`:
            [
                vector of token,
                vector of token,
                vector of token
            ]
        '''
        raise NotImplementedError("This method must be implemented.")
