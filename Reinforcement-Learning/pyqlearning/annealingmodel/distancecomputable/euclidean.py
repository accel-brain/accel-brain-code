# -*- coding: utf-8 -*-
from pyqlearning.annealingmodel.distance_computable import DistanceComputable
import numpy as np


class Euclidean(DistanceComputable):
    '''
    Distance.
    '''

    def compute(self, x, y):
        '''
        Args:
            x:    Data point.
            y:    Data point.
        
        Returns:
            Distance.
        '''
        return np.sqrt(np.sum((x-y)**2))
