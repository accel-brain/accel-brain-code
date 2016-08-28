#!/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from deeplearning.activation.interface.activating_function_interface import ActivatingFunctionInterface


class LogisticFunction(ActivatingFunctionInterface):
    '''
    ロジスティック関数
    '''

    def activate(self, x):
        '''
        活性化関数の返り値を返す

        Args:
            x   パラメタ

        Returns:
            活性化関数の返り値
        '''
        return 1.0 / (1.0 + np.exp(-x))
