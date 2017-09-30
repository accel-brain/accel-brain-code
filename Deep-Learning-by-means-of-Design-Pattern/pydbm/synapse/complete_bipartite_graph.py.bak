#!/user/bin/env python
# -*- coding: utf-8 -*-
from deeplearning.synapse_list import Synapse


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
        for i in range(len(self.visible_neuron_list)):
            for j in range(len(self.hidden_neuron_list)):
                self.diff_weights_dict[(i, j)] = self.__update_weight(
                    self.visible_neuron_list[i].activity,
                    self.hidden_neuron_list[j].activity,
                    learning_rate
                )

    def __update_weight(self, visible_activity, hidden_activity, learning_rate):
        '''
        各リンクの重みの更新値を計算する

        Args:
            visible_activity:   可視層ニューロンの活性度
            hidden_activity:    隠れ層のニューロンの活性度
            learning_rate:      学習率

        Returns:
            重み
        '''
        weight = visible_activity * hidden_activity * learning_rate
        return weight
