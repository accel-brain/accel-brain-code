# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class DBMBuilder(metaclass=ABCMeta):
    '''
    GoFのデザイン・パタンの「Builder Pattern」の「建築者」のインターフェイス
    制限ボルツマンマシンを組み立てることで、
    深層ボルツマンマシンのオブジェクトを生成する

    抽象メソッドのみの抽象クラスを便宜上インターフェイスとして扱う
    '''

    @abstractproperty
    def learning_rate(self):
        ''' 学習率 '''
        raise NotImplementedError()

    @abstractmethod
    def visible_neuron_part(self, activating_function, neuron_count):
        '''
        可視層ニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError()

    @abstractmethod
    def feature_neuron_part(self, activating_function, neuron_count):
        '''
        疑似的な可視層ニューロンとして振る舞う隠れ層ニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError()

    @abstractmethod
    def hidden_neuron_part(self, activating_function, neuron_count):
        '''
        隠れ層ニューロンを生成する

        Args:
            activation_function:    活性化関数
            neuron_count:           生成するニューロンの個数
        '''
        raise NotImplementedError

    @abstractmethod
    def graph_part(self, approximate_interface):
        '''
        完全二部グラフを生成する

        Args:
            approximate_interface:    近似用のオブジェクト
        '''
        raise NotImplementedError()

    @abstractmethod
    def get_result():
        '''
        「Builder Pattern」によって生成された一連のオブジェクトを返す

        Returns:
            制限ボルツマンマシンのリスト
        '''
        raise NotImplementedError()
