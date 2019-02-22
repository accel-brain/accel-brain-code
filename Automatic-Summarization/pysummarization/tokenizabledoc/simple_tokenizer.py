# -*- coding: utf-8 -*-
from pysummarization.tokenizable_doc import TokenizableDoc


class SimpleTokenizer(TokenizableDoc):
    '''
    Tokenize delimited sentence with a blank.
    '''
    
    def tokenize(self, sentence_str):
        '''
        Tokenize str.
        
        Args:
            sentence_str:   tokenized string.
        
        Returns:
            [token, token, token, ...]
        '''
        token_list = sentence_str.split(" ")
        return token_list
