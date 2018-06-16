# -*- coding: utf-8 -*-
from pyqlearning.annealingmodel.distance_computable import DistanceComputable
from pyqlearning.annealingmodel.cost_functionable import CostFunctionable
import numpy as np


class CostAsDistance(DistanceComputable):
    '''
    Compute cost as distance.
    '''

    # Memo of parameters.
    __memo_dict = {}

    def __init__(self, params_arr, cost_functionable):
        '''
        Init.
        
        Args:
            params_arr:           The parameters.
            cost_functionable:    is-a `CostFunctionable`.
        
        '''
        self.__params_arr = params_arr
        if isinstance(cost_functionable, CostFunctionable):
            self.__cost_functionable = cost_functionable
        else:
            raise TypeError

    def compute(self, x, y):
        '''
        Compute distance.
        
        Args:
            x:    Data point.
            y:    Data point.
        
        Returns:
            Distance.
        '''
        if x in self.__memo_dict:
            x_v = self.__memo_dict[x]
        else:
            x_v = self.__cost_functionable.compute(self.__params_arr[x, :])
            self.__memo_dict.setdefault(x, x_v)
        if y in self.__memo_dict:
            y_v = self.__memo_dict[y]
        else:
            y_v = self.__cost_functionable.compute(self.__params_arr[y, :])
            self.__memo_dict.setdefault(y, y_v)

        return abs(x_v - y_v)
