# -*- coding: utf-8 -*-
from pysummarization.similarity_filter import SimilarityFilter


class Tanimoto(SimilarityFilter):
    '''
    Concrete class for filtering mutually similar sentences.
    '''

    def calculate(self, token_list_x, token_list_y):
        '''
        Calculate similarity with the Tanimoto coefficient.
        
        Concrete method.
        
        Args:
            token_list_x:    [token, token, token, ...]
            token_list_y:    [token, token, token, ...]
        
        Returns:
            Similarity.
        '''
        match_list = [tanimoto_value for tanimoto_value in token_list_x if tanimoto_value in token_list_y]
        return float(len(match_list) / (len(token_list_x) + len(token_list_y) - len(match_list)))
