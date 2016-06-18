#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class OutputLayerInterface(metaclass=ABCMeta):
    '''
    出力層の学習を実行させるためのインターフェイス
    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractmethod
    def output_update_state(self, link_value):
        '''
        出力層の学習

        Args:
            link_value:      リンク先による入力値

        '''
        raise NotImplementedError()

    @abstractmethod
    def release(self):
        '''
        活性度を放出する

        Returns:
            活性度
        '''
        raise NotImplementedError()
