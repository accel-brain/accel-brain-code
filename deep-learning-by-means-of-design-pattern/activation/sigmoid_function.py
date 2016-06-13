#!/user/bin/env python
# -*- coding: utf-8 -*-
import math
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface


class SigmoidFunction(ActivatingFunctionInterface):
    '''
    ジグモイド関数

    #TODO(chimera0):オーバーフロー対策をnumpyなどの既存関数で実施する
    '''

    # オーバーフロー対策
    __limit = -709

    def activate(self, x):
        '''
        活性化関数の返り値を返す

        Args:
            x   パラメタ

        Returns:
            活性化関数の返り値
        '''
        if x >= self.__limit:
            y = 1.0 / (1.0 + math.exp((-1) * x))
        else:
            y = 0.0
        
        return y
