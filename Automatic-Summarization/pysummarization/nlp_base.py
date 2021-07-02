# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pysummarization.tokenizable_doc import TokenizableDoc


class NlpBase(object):
    '''
    The base class for NLP.
    '''
    
    # object of tokenizer.
    __tokenizable_doc = None

    def get_tokenizable_doc(self):
        ''' getter '''
        if isinstance(self.__tokenizable_doc, TokenizableDoc):
            return self.__tokenizable_doc
        else:
            raise TypeError()
    
    def set_tokenizable_doc(self, value):
        ''' setter '''
        if isinstance(value, TokenizableDoc):
            self.__tokenizable_doc = value
        else:
            raise TypeError()

    tokenizable_doc = property(get_tokenizable_doc, set_tokenizable_doc)

    # Delimiter for self.listup_sentence.
    __delimiter_list = ["。", "\n", ".", " ", "　", "．"]
    
    def get_delimiter_list(self):
        ''' getter '''
        return self.__delimiter_list

    def set_delimiter_list(self, value):
        ''' setter '''
        self.__delimiter_list = value

    delimiter_list = property(get_delimiter_list, set_delimiter_list)

    # List of tokens.
    __token = []

    def get_token(self):
        ''' getter '''
        return self.__token

    def set_token(self, value):
        ''' setter '''
        self.__token = value

    token = property(get_token, set_token)

    def __init__(self):
        ''' Init. '''
        self.__token = []
        self.__delimiter_list = ["。", "\n", ".", " ", "　", "．"]
        self.__tokenizable_doc = None

    def tokenize(self, data):
        '''
        Tokenize sentence and set the list of tokens to self.token.

        Args:
            data:    string.

        '''
        self.token = self.tokenizable_doc.tokenize(data)

    def listup_sentence(self, data, counter=0):
        '''
        Divide string into sentence list.

        Args:
            data:               string.
            counter:            recursive counter.

        Returns:
            List of sentences.

        '''
        delimiter = self.delimiter_list[counter]
        sentence_list = []
        [sentence_list.append(sentence + delimiter) for sentence in data.split(delimiter) if sentence != ""]
        if counter + 1 < len(self.delimiter_list):
            sentence_list_r = []
            [sentence_list_r.extend(self.listup_sentence(sentence, counter+1)) for sentence in sentence_list]
            sentence_list = sentence_list_r

        return sentence_list
