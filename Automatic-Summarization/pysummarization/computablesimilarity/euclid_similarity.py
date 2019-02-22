# -*- coding: utf-8 -*-
from pysummarization.computable_similarity import ComputableSimilarity
import numpy as np


class EuclidSimilarity(ComputableSimilarity):
    '''
    Compute Euclid similarity between two vectors of sentences.
    '''

    def __init__(self, min_max_norm_flag=True):
        '''
        Init.

        Args:
            min_max_norm_flag:      if `True`, the computed similarity is normalized by min-max method.
        '''
        self.__min_max_norm_flag = min_max_norm_flag

    def compute(self, vector_x_arr, vector_y_arr):
        '''
        Compute similarity.

        Args:
            vector_x_arr:   `np.ndarray` of vectors.
            vector_y_arr:   `np.ndarray` of vectors.
        
        Returns:
            `np.ndarray` of similarities.
        '''
        distance_arr = np.sqrt(np.power(vector_x_arr - vector_y_arr, 2))
        if self.__min_max_norm_flag is False:
            return 1 / distance_arr
        else:
            if distance_arr.shape[0] > 1 and distance_arr.max() > distance_arr.min():
                distance_arr = (distance_arr - distance_arr.min()) / (distance_arr.max() - distance_arr.min())
                return 1 - distance_arr
            else:
                return 1 / distance_arr
