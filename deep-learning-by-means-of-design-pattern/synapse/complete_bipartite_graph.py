#!/user/bin/env python
# -*- coding: utf-8 -*-
import sys

from deeplearning.synapse_list import Synapse


class CompleteBipartiteGraph(Synapse):
    '''
    シナプスを有したニューラル・ネットワークとしての完全二部グラフ
    '''

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
