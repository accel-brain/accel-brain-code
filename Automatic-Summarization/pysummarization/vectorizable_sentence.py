# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class VectorizableSentence(metaclass=ABCMeta):
    '''
    Vectorize sentence.
    '''
    
    @abstractmethod
    def vectorize(self, sentence_list):
        '''
        Tokenize token list.
        
        Args:
            sentence_list:   The list of tokenized sentences.
                             [[`token`, `token`, `token`, ...],
                             [`token`, `token`, `token`, ...],
                             [`token`, `token`, `token`, ...]]
        
        Returns:
            `np.ndarray` of tokens.
            [vector of token, vector of token, vector of token]
        '''
        raise NotImplementedError("This method must be implemented.")
