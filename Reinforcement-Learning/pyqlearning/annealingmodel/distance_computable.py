# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class DistanceComputable(metaclass=ABCMeta):
    '''
    The interface of computing distance.
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
