# -*- coding: utf-8 -*-
from pysummarization.computable_similarity import ComputableSimilarity
import numpy as np


class CosineSimilarity(ComputableSimilarity):
    '''
    Compute cosine similarity between two vectors of sentences.
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
        if vector_x_arr.ndim != vector_y_arr.ndim:
            raise ValueError("`vector_x_arr.ndim` != `vector_y_arr.ndim`.")

        if vector_x_arr.ndim == 1 and vector_y_arr.ndim == 1:
            return self.__cosine(vector_x_arr, vector_y_arr)
        else:
            result_arr = np.zeros(vector_x_arr.shape[0])
            for i in range(vector_x_arr.shape[0]):
                if vector_x_arr[i].ndim > 1:
                    result_arr[i] = self.__cosine(
                        vector_x_arr[i].reshape(-1, 1),
                        vector_y_arr[i].reshape(-1, 1)
                    )
                else:
                    result_arr[i] = self.__cosine(vector_x_arr[i], vector_y_arr[i])
            return result_arr

    def __cosine(self, x_arr, y_arr):
        try:
            return np.dot(x_arr, y_arr) / np.linalg.norm(x_arr) * np.linalg.norm(y_arr)
        except ZeroDivisionError:
            return np.inf
