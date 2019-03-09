# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.computable_distance import ComputableDistance


class KLDivergence(ComputableDistance):
    '''
    Compute Kullback-Leibler divergence(KLD) between two vectors.

    This class considers KLD as a kind of distance.
    '''

    def compute(self, x_arr, y_arr):
        '''
        Compute distance.

        Args:
            x_arr:      `np.ndarray` of vectors.
            y_arr:      `np.ndarray` of vectors.

        Retruns:
            `np.ndarray` of distances.
        '''
        y_arr += 1e-08
        return np.sum(x_arr * np.log(x_arr / y_arr), axis=-1)
