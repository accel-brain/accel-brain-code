# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.iter_text_generator import IterTextGenerator


class IterSentenceGenerator(IterTextGenerator):
    '''
    Iterator/Generator that generates vectors of sentences.
    '''

    def generate_uniform(self):
        '''
        Draw the samples from uniform distribution.

        Returns:
            Tuple data.
            - `np.ndarray` of vectors.
            - `np.ndarray` of tokens.
        '''
        for vector_arr, token_arr in super().generate_uniform():
            vector_arr = vector_arr.reshape((
                vector_arr.shape[0],
                1,
                vector_arr.shape[1],
                vector_arr.shape[2]
            ))
            yield vector_arr, token_arr

    def generate_real_token(self):
        '''
        Draw the samples from real sentences.

        Returns:
            Tuple data.
            - `np.ndarray` of vectors.
            - `np.ndarray` of tokens.
        '''
        for vector_arr, token_arr in super().generate_real_token():
            vector_arr = vector_arr.reshape((
                vector_arr.shape[0],
                1,
                vector_arr.shape[1],
                vector_arr.shape[2]
            ))
            yield vector_arr, token_arr
