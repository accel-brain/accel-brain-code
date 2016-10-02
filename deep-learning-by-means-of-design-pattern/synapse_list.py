#!/user/bin/env python
# -*- coding: utf-8 -*-
import random
from accelstat.normalization import Normalization


class Synapse(object):
    '''
    抽象クラス：Neuronの下位クラスの関連を保持するシナプス
    '''

    # 比較的浅い層のニューロンオブジェクトリスト
    __shallower_neuron_list = []
    # 比較的深い層のニューロンオブジェクトリスト
    __deeper_neuron_list = []
    # 各リンクの重みリスト
    __weights_dict = {}
    # 各リンクの重みの差分リスト
    __diff_weights_dict = {}

    def get_shallower_neuron_list(self):
        ''' getter '''
        return self.__shallower_neuron_list

    def set_shallower_neuron_list(self, value):
        ''' setter '''
        self.__shallower_neuron_list = value

    def get_deeper_neuron_list(self):
        ''' getter '''
        return self.__deeper_neuron_list

    def set_deeper_neuron_list(self, value):
        ''' setter '''
        self.__deeper_neuron_list = value

    def get_weights_dict(self):
        ''' getter '''
        return self.__weights_dict

    def set_weights_dict(self, value):
        ''' setter '''
        self.__weights_dict = value

    def get_diff_weights_dict(self):
        ''' getter '''
        return self.__diff_weights_dict

    def set_diff_weights_dict(self, value):
        ''' setter '''
        self.__diff_weights_dict = value

    shallower_neuron_list = property(get_shallower_neuron_list, set_shallower_neuron_list)
    deeper_neuron_list = property(get_deeper_neuron_list, set_deeper_neuron_list)
    weights_dict = property(get_weights_dict, set_weights_dict)
    diff_weights_dict = property(get_diff_weights_dict, set_diff_weights_dict)

    # 規格化オブジェクト
    __normalization = None

    def create_node(self, shallower_neuron_list, deeper_neuron_list, weights_dict=None):
        '''
        グラフにノードのリンクをセットする

        Args:
            shallower_neuron_list:    可視層のニューロンオブジェクトリスト
            deeper_neuron_list:     隠れ層のニューロンオブジェクトリスト
            weights_dict:           各リンクの重みリスト
        '''
        self.__shallower_neuron_list = shallower_neuron_list
        self.__deeper_neuron_list = deeper_neuron_list
        if weights_dict is not None:
            self.__weights_dict = weights_dict
        else:
            for i in range(len(shallower_neuron_list)):
                for j in range(len(deeper_neuron_list)):
                    self.__weights_dict[(i, j)] = round(random.random(), 3)

    def learn_weights(self):
        '''
        リンクの重みを更新する
        '''
        for i, j in self.diff_weights_dict:
            self.weights_dict[(i, j)] += self.diff_weights_dict[(i, j)]
            self.diff_weights_dict[(i, j)] = 0.0

    def normalize_bias_and_weights(self):
        '''
        規格化する
        '''
        if self.__normalization is None:
            self.__normalization = Normalization()

        visible_activity_list = [self.shallower_neuron_list[i].activity for i in range(len(self.shallower_neuron_list))]
        hidden_activity_list = [self.deeper_neuron_list[i].activity for i in range(len(self.deeper_neuron_list))]

        visible_activity_list = self.__normalization.z_theta(visible_activity_list)
        hidden_activity_list = self.__normalization.z_theta(hidden_activity_list)

        for i in range(len(visible_activity_list)):
            self.shallower_neuron_list[i].activity = visible_activity_list[i]

        for i in range(len(hidden_activity_list)):
            self.deeper_neuron_list[i].activity = hidden_activity_list[i]

        weights_list = []
        for i in range(len(self.shallower_neuron_list)):
            for j in range(len(self.deeper_neuron_list)):
                weights_list.append(self.weights_dict[(i, j)])

        for i in range(len(self.shallower_neuron_list)):
            for j in range(len(self.deeper_neuron_list)):
                self.weights_dict[(i, j)] = self.__normalization.once(
                    x=self.weights_dict[(i, j)],
                    x_min=min(weights_list),
                    x_max=max(weights_list),
                    range_min=0.0,
                    range_max=1.0
            )
