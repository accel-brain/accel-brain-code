#!/user/bin/env python
# -*- coding: utf-8 -*-
import math
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface


class SigmoidFunction(ActivatingFunctionInterface):
    '''
    ジグモイド関数

    #TODO(chimera0):オーバーフロー対策をnumpyなどの既存関数で実施する
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
