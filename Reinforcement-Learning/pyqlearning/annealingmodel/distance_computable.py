# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class DistanceComputable(metaclass=ABCMeta):
    '''
    The interface of computing distance,
    to compute cost as distance for `QuantumMonteCarlo`.
    
    References:
        - Das, A., & Chakrabarti, B. K. (Eds.). (2005). Quantum annealing and related optimization methods (Vol. 679). Springer Science & Business Media.

    '''
    
    def compute(self, x, y):
        '''
        Compute distance.
        
        Args:
            x:    Data point.
            y:    Data point.
        
        Returns:
            Distance.
        '''
        raise NotImplementedError()
