# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ComputableSimilarity(metaclass=ABCMeta):
    '''
    Compute similarity between two vectors of sentences.
    '''
    
    def compute(self, vector_x_arr, vector_y_arr):
        '''
        Compute similarity.

        Args:
            vector_x_arr:   `np.ndarray` of vectors.
            vector_y_arr:   `np.ndarray` of vectors.
        
        Returns:
            `np.ndarray` of similarities.
        
        '''
        raise NotImplementedError()
