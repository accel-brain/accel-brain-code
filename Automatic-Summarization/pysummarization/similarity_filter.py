# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pysummarization.nlp_base import NlpBase


class SimilarityFilter(metaclass=ABCMeta):
    '''
    Abstract class for filtering mutually similar sentences.
    '''

    # NlpBase
    __nlp_base = None

    def get_nlp_base(self):
        ''' getter '''
        if isinstance(self.__nlp_base, NlpBase) is False:
            raise TypeError("The type of self.__nlp_base must be NlpBase.")

        return self.__nlp_base

    def set_nlp_base(self, value):
        ''' setter '''
        if isinstance(value, NlpBase) is False:
            raise TypeError("The type of value must be NlpBase.")
        
        self.__nlp_base = value

    nlp_base = property(get_nlp_base, set_nlp_base)

    # Cut off threshold.
    __similarity_limit = 0.8

    def get_similarity_limit(self):
        ''' getter '''
        if isinstance(self.__similarity_limit, float) is False:
            raise TypeError("__similarity_limit must be float.")
        return self.__similarity_limit

    def set_similarity_limit(self, value):
        ''' setter '''
        if isinstance(value, float) is False:
            raise TypeError("__similarity_limit must be float.")
        self.__similarity_limit = value

    similarity_limit = property(get_similarity_limit, set_similarity_limit)

    @abstractmethod
    def calculate(self, token_list_x, token_list_y):
        '''
        Calculate similarity.
        
        Abstract method.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]
        
        Returns:
            Similarity.
        '''
        raise NotImplementedError("This method must be implemented.")

    def unique(self, token_list_x, token_list_y):
        '''
        Remove duplicated elements.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]

        Returns:
            Tuple(token_list_x, token_list_y)
        '''
        x = set(list(token_list_x))
        y = set(list(token_list_y))
        return (x, y)

    def count(self, token_list):
        '''
        Count the number of tokens in `token_list`.
        
        Args:
            token_list:    The list of tokens.

        Returns:
            {token: the numbers}
        '''
        token_dict = {}
        for token in token_list:
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1
        return token_dict

    def similar_filter_r(self, sentence_list):
        '''
        Filter mutually similar sentences.
        
        Args:
            sentence_list:    The list of sentences.

        Returns:
            The list of filtered sentences.
        '''
        result_list = []
        recursive_list = []

        try:
            self.nlp_base.tokenize(sentence_list[0])
            subject_token = self.nlp_base.token
            result_list.append(sentence_list[0])
            if len(sentence_list) > 1:
                for i in range(len(sentence_list)):
                    if i > 0:
                        self.nlp_base.tokenize(sentence_list[i])
                        object_token = self.nlp_base.token
                        similarity = self.calculate(subject_token, object_token)
                        if similarity <= self.similarity_limit:
                            recursive_list.append(sentence_list[i])

            if len(recursive_list) > 0:
                result_list.extend(self.similar_filter_r(recursive_list))
        except IndexError:
            result_list = sentence_list

        return result_list
