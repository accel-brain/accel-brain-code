#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class NNBuilder(metaclass=ABCMeta):
    '''
    GoFのデザイン・パタンの「Builder Pattern」の「建築者」のインターフェイス
    パーセプトロンのシナプス部分を組み立てることで、
    ニューラルネットワークのオブジェクトを生成する

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractproperty
    def learning_rate(self):
        ''' 学習率 '''
        raise NotImplementedError()

    @abstractmethod
    def input_neuron_part(self, activating_function, neuron_count):
        '''
        入力層ニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        中間層（隠れ層）のニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError()

    @abstractmethod
    def output_neuron_part(self, activating_function, neuron_count):
        '''
        出力層ニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError

    @abstractmethod
    def graph_part(self, approximate_interface):
        '''
        シナプスのグラフを生成する

        Args:
            approximate_interface:    近似用のオブジェクト
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_result():
        '''
        「Builder Pattern」によって生成された一連のオブジェクトを返す

        Returns:
            シナプスのリスト
        '''
        raise NotImplementedError()
