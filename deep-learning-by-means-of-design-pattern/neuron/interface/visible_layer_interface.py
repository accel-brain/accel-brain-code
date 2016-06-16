#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class VisibleLayerInterface(metaclass=ABCMeta):
    '''
    可視層の学習を実行させるためのインターフェイス
    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractmethod
    def observe_data_point(self, x):
        '''
        観測データ点の入力

        Args:
            x:  観測データ点
        '''
        raise NotImplementedError()

    @abstractmethod
    def visible_update_state(self, link_value):
        '''
        可視層の学習

        Args:
            link_value:      リンク先による入力値

        '''
        raise NotImplementedError()
