# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ComputableDistance(metaclass=ABCMeta):
    '''
    Compute distances between two vectors.
    '''
    
    @abstractmethod
    def compute(self, x_arr, y_arr):
        '''
        Compute distance.

        Args:
            x_arr:      `np.ndarray` of vectors.
            y_arr:      `np.ndarray` of vectors.

        Retruns:
            `np.ndarray` of distances.
        '''
        raise NotImplementedError("This method must be implemented.")
