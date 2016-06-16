#!/user/bin/env python
# -*- coding: utf-8 -*-
import random
from deeplearning.neuron_object import Neuron
from deeplearning.neuron.interface.hidden_layer_interface import HiddenLayerInterface


class HiddenNeuron(Neuron, HiddenLayerInterface):
    '''
    隠れ層のニューロン
    '''

    def __init__(self):
        self.bias = round(random.random(), 3)

    def hidden_update_state(self, link_value):
        '''
        隠れ層の学習

        Args:
            link_value:      リンク先による入力値

        '''
        # 活性度の判定
        output = self.activate(link_value)
        self.activity = output

    def update_bias(self, learning_rate):
        '''
        バイアスの調整

        Args:
            learning_rate:  学習率
        '''
        self.diff_bias += learning_rate * self.activity
