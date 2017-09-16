#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ActivatingFunctionInterface(metaclass=ABCMeta):
    '''
    活性化関数をニューロンオブジェクトに委譲するための
    インターフェイス

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractmethod
    def activate(self, x):
        '''
        活性化関数の返り値を返す

        Args:
            x:   パラメタ

        Returns:
            活性化関数の返り値
        '''
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, y):
        '''
        導関数

        Args:
            y:  パラメタ
        Returns:
            導関数の値
        '''
        raise NotImplementedError()
