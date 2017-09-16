#!/user/bin/env python
# -*- coding: utf-8 -*-
import math
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface


class TanhFunction(ActivatingFunctionInterface):
    '''
    tanh関数
    '''

    def activate(self, x):
        '''
        活性化関数の返り値を返す

        Args:
            x   パラメタ

        Returns:
            活性化関数の返り値
        '''
        return math.tanh(x)

    def derivative(self, y):
        '''
        導関数

        Args:
            y:  パラメタ
        Returns:
            導関数の値
        '''
        return 1.0 - y**2
