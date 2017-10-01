#!/user/bin/env python
# -*- coding: utf-8 -*-
import pyximport;pyximport.install()
import random
from pydbm.neuron_object import Neuron
from pydbm.neuron.interface.visible_layer_interface import VisibleLayerInterface
from pydbm.neuron.interface.hidden_layer_interface import HiddenLayerInterface
from pydbm.activation.interface.activating_function_interface import ActivatingFunctionInterface


class FeaturePointNeuron(Neuron, VisibleLayerInterface, HiddenLayerInterface):
    '''
    深層ボルツマンマシンの特徴点を疑似的な観測データ点と見立てることを前提に、
    隠れ層のニューロンを疑似的な可視層ニューロンとしてインスタンス化するクラス

    隠れ層のニューロンの機能的等価物であると共に、
    可視層のニューロンの機能的等価物としても実装する
    '''

    # 可視層のニューロン
    __visible_layer_interface = None

    def get_activating_function(self):
        ''' getter of activating_function '''
        if isinstance(self.__activating_function, ActivatingFunctionInterface) is False:
            raise TypeError()
        return self.__visible_layer_interface.activating_function

    def set_activating_function(self, value):
        ''' setter of activating_function '''
        if isinstance(value, ActivatingFunctionInterface) is False:
            raise TypeError()
        self.__visible_layer_interface.activating_function = value

    activating_function = property(get_activating_function, set_activating_function)

    def __init__(self, visible_layer_interface):
        '''
        初期化
        自分自身をvisible_layer_interfaceの型で仕様化すると共に、
        本来のvisible_layer_interfaceを実現した可視層ニューロンを包含することで、
        自分自身も可視層ニューロンとして振る舞えるようにする

        Args:
            visible_layer_interface:    隠れ層のニューロンのインターフェイスを実装したニューロンオブジェクト
        '''
        self.bias = round(random.random(), 3)
        visible_layer_interface.bias = self.bias
        self.__visible_layer_interface = visible_layer_interface

    def observe_data_point(self, double x):
        '''
        インターフェイス実現
        観測データ点の入力

        Args:
            x:  観測データ点
        '''
        self.__visible_layer_interface.observe_data_point(x)
        self.activity = x

    def visible_update_state(self, double link_value):
        '''
        インターフェイス実現
        可視層ニューロンとしての学習

        Args:
            link_value:      リンク先による入力値

        '''
        self.__visible_layer_interface.visible_update_state(link_value)
        self.activity = self.__visible_layer_interface.activity

    def hidden_update_state(self, double link_value):
        '''
        インターフェイス実現
        隠れ層ニューロンとしての学習

        Args:
            link_value:      リンク先による入力値

        '''
        self.visible_update_state(link_value)

    def update_bias(self, double learning_rate):
        '''
        具象メソッド
        バイアスの調整

        Args:
            learning_rate:  学習率
        '''
        diff_bias = learning_rate * self.activity
        self.__visible_layer_interface.diff_bias += diff_bias
        self.diff_bias += diff_bias
