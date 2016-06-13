#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class HiddenLayerInterface(metaclass=ABCMeta):
    '''
    隠れ層の学習を実行させるためのインターフェイス

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractmethod
    def hidden_update_state(self, weight, link_value):
        '''
        可視層の学習

        Args:
            weight:          重み
            link_value:      リンク先による入力値

        '''
        raise NotImplementedError()
