# -*- coding: utf-8 -*-
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from pydbm.synapse_list import Synapse


class CompleteBipartiteGraph(Synapse):
    '''
    シナプスを有したニューラル・ネットワークとしての完全二部グラフ
    '''

    def get_visible_neuron_list(self):
        '''
        比較的浅い層を可視層と見立てる
        '''
        return self.shallower_neuron_list

    def set_visible_neuron_list(self, value):
        '''
        比較的浅い層を可視層と見立てる
        '''
        self.shallower_neuron_list = value

    visible_neuron_list = property(get_visible_neuron_list, set_visible_neuron_list)

    def get_hidden_neuron_list(self):
        '''
        比較的深い層を隠れ層と見立てる
        '''
        return self.deeper_neuron_list

    def set_hidden_neuron_list(self, value):
        '''
        比較的深い層を隠れ層と見立てる
        '''
        self.deeper_neuron_list = value

    hidden_neuron_list = property(get_hidden_neuron_list, set_hidden_neuron_list)
    
    def update(self, learning_rate):
        '''
        各リンクの重みを更新する

        Args:
            learning_rate:  学習率
        '''
        visible_activity_arr = np.array([[self.visible_neuron_list[i].activity] * len(self.hidden_neuron_list) for i in range(len(self.visible_neuron_list))])
        hidden_activity_arr = np.array([[self.hidden_neuron_list[j].activity] * len(self.visible_neuron_list) for j in range(len(self.hidden_neuron_list))])
        self.diff_weights_arr = visible_activity_arr * hidden_activity_arr.T * learning_rate
