#!/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


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

    def derivative(self, y):
        '''
        導関数

        Args:
            y:  パラメタ
        Returns:
            導関数の値
        '''
        return self.activate(y) * (1 - self.activate(y))
