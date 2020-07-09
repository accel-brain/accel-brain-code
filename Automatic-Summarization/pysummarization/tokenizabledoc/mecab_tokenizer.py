# -*- coding: utf-8 -*-
from pysummarization.tokenizable_doc import TokenizableDoc
import MeCab


class MeCabTokenizer(TokenizableDoc):
    '''
    Tokenize string.
    
    Japanese morphological analysis with MeCab.
    '''

    __part_of_speech = ["名詞", "形容詞", "動詞"]

    def get_part_of_speech(self):
        ''' getter '''
        return self.__part_of_speech
    
    def set_part_of_speech(self, value):
        ''' setter '''
        self.__part_of_speech = value

    part_of_speech = property(get_part_of_speech, set_part_of_speech)

    def tokenize(self, sentence_str):
        '''
        Tokenize str.
        
        Args:
            sentence_str:   tokenized string.
        
        Returns:
            [token, token, token, ...]
        '''
        if len(self.part_of_speech) == 0:
            mt = MeCab.Tagger("-Owakati")
            wordlist = mt.parse(sentence_str)
            token_list = wordlist.rstrip(" \n").split(" ")
            return token_list
        else:
            tagger = MeCab.Tagger(" -Ochasen")
            node = tagger.parseToNode(sentence_str)

            result_tuple_list = []
            token_list = []
            while node:
                feature_list = node.feature.split(",")
                if feature_list[0] != "BOS/EOS":
                    if feature_list[0] in self.part_of_speech:
                        token = feature_list[6]
                        token_list.append(token)
                node = node.next
            return token_list
