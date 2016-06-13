#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ApproximateInterface(metaclass=ABCMeta):
    '''
    近似学習用インターフェイス

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractmethod
    def approximate_learn(self, graph, learning_rate, observed_data_matrix, traning_count=1000):
        '''
        近似による学習

        Args:
            graph:                ニューロンのグラフ
            learning_rate:        学習率
            observed_data_matrix: 観測データ点
            traning_count:        訓練回数

        '''
        raise NotImplementedError()
