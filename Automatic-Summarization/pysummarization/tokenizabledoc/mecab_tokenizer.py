# -*- coding: utf-8 -*-
from pysummarization.tokenizable_doc import TokenizableDoc
import MeCab


class MeCabTokenizer(TokenizableDoc):
    '''
    Tokenize string.
    
    Japanese morphological analysis with MeCab.
    '''
    
    def tokenize(self, sentence_str):
        '''
        Tokenize str.
        
        Args:
            sentence_str:   tokenized string.
        
        Returns:
            [token, token, token, ...]
        '''
        mt = MeCab.Tagger("-Owakati")
        wordlist = mt.parse(sentence_str)
        token_list = wordlist.rstrip(" \n").split(" ")
        return token_list
