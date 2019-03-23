# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.computable_distance import ComputableDistance


class EuclidDistance(ComputableDistance):
    '''
    Compute Euclid distances between two vectors.
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
        return np.linalg.norm(x_arr - y_arr, axis=-1)
