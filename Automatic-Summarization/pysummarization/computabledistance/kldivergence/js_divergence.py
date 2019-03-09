# -*- coding: utf-8 -*-
import numpy as np
from pysummarization.computabledistance.kl_divergence import KLDivergence


class JSDivergence(KLDivergence):
    '''
    Compute Jensen-Shannon divergence(JSD) between two vectors.

    This class considers JSD as a kind of distance.
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
        x_arr = x_arr / np.linalg.norm(x_arr, ord=1)
        y_arr = y_arr / np.linalg.norm(y_arr, ord=1)
        mixture_arr = 0.5 * (x_arr + y_arr)
        return 0.5 * (super().compute(x_arr, mixture_arr) + super().compute(y_arr, mixture_arr))
