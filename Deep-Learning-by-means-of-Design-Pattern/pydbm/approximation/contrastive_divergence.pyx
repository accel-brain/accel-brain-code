#!/user/bin/env python
# -*- coding: utf-8 -*-
import pyximport
import numpy as np
pyximport.install(setup_args={'include_dirs':[np.get_include()]}, inplace=True)
from pydbm.approximation.interface.approximate_interface import ApproximateInterface


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence法
    Numpy版

    概念的にはpositive phaseとnegative phaseの区別を
    wakeとsleepの区別に対応付けて各メソッドを命名している。

    '''

    # ニューロンのグラフ
    __graph = None
    # 学習率
    __learning_rate = 0.5

    def approximate_learn(self, graph, double learning_rate, observed_data_matrix, int traning_count=1000):
        '''
        インターフェイスの実現
        近似による学習
        目覚めと眠りのフェーズを逐次実行して学習する

        Args:
            graph:                ニューロンのグラフ
            learning_rate:        学習率
            observed_data_matrix: 観測データ点
            traning_count:        訓練回数

        Returns:
            グラフ
        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        [self.__wake_and_sleep(observed_data_list) for observed_data_list in observed_data_matrix]
        return self.__graph

    def __wake_and_sleep(self, observed_data_list):
        '''
        目覚めと夢見のリズム

        Args:
            observed_data_list:      観測データ点
        '''
        self.__wake(observed_data_list)
        self.__sleep()
        self.__learn()

    def __wake(self, observed_data_list):
        '''
        目覚めと眠りのフェーズを逐次実行して学習する

        Args:
            observed_data_list:      観測データ点
        '''
        # 観測データ点を可視ニューロンに入力する
        cdef int k
        [self.__graph.visible_neuron_list[k].observe_data_point(observed_data_list[k]) for k in range(len(observed_data_list))]
        # 隠れ層のニューロンの発火状態を更新する
        self.__update_hidden_spike()
        # ヘブ規則によりリンクの重みを更新する
        self.__graph.update(self.__learning_rate)
        # 可視層のバイアスを更新する
        cdef int i
        [self.__graph.visible_neuron_list[i].update_bias(self.__learning_rate) for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスを更新する
        cdef int j
        [self.__graph.hidden_neuron_list[j].update_bias(self.__learning_rate) for j in range(len(self.__graph.hidden_neuron_list))]

    def __sleep(self):
        '''
        夢を見る
        自由連想
        '''
        # 可視層のニューロンの発火状態を更新する
        self.__update_visible_spike()
        # 隠れ層のニューロンの発火状態を更新する
        self.__update_hidden_spike()
        # ヘブ規則によりリンクの重みを逆更新する
        self.__graph.update((-1) * self.__learning_rate)
        # 可視層のバイアスを更新する
        cdef int i
        [self.__graph.visible_neuron_list[i].update_bias((-1) * self.__learning_rate) for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスを更新する
        cdef int j
        [self.__graph.hidden_neuron_list[j].update_bias((-1) * self.__learning_rate) for j in range(len(self.__graph.hidden_neuron_list))]

    def __update_visible_spike(self):
        '''
        可視層のニューロンの発火状態を更新する
        '''
        hidden_activity_arr = np.array([[self.__graph.hidden_neuron_list[j].activity] * self.__graph.weights_arr.T.shape[1] for j in range(len(self.__graph.hidden_neuron_list))])
        link_value_arr = self.__graph.weights_arr.T * hidden_activity_arr
        visible_activity_arr = link_value_arr.sum(axis=0)
        cdef int i
        for i in range(visible_activity_arr.shape[0]):
            self.__graph.visible_neuron_list[i].visible_update_state(visible_activity_arr[i])
            self.__graph.normalize_visible_bias()

    def __update_hidden_spike(self):
        '''
        隠れ層のニューロンの発火状態を更新する
        '''
        visible_activity_arr = np.array([[self.__graph.visible_neuron_list[i].activity] * self.__graph.weights_arr.shape[1] for i in range(len(self.__graph.visible_neuron_list))])
        link_value_arr = self.__graph.weights_arr * visible_activity_arr
        hidden_activity_arr = link_value_arr.sum(axis=0)
        cdef int j
        for j in range(hidden_activity_arr.shape[0]):
            self.__graph.hidden_neuron_list[j].hidden_update_state(hidden_activity_arr[j])
            self.__graph.normalize_hidden_bias()

    def __learn(self):
        '''
        バイアスと重みを学習する
        '''
        # 可視層のバイアスの学習
        cdef int i
        [self.__graph.visible_neuron_list[i].learn_bias() for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスの学習
        cdef int j
        [self.__graph.hidden_neuron_list[j].learn_bias() for j in range(len(self.__graph.hidden_neuron_list))]
        # リンクの重みの学習
        self.__graph.learn_weights()

    def recall(self, graph, observed_data_matrix):
        '''
        記憶から自由連想する

        Args:
            graph:                  自由連想前のグラフ
            observed_data_matrix:   観測データ点

        Returns:
            自由連想後のグラフ

        '''
        self.__graph = graph
        cdef int k
        [self.__wake_and_sleep(observed_data_matrix[k]) for k in range(len(observed_data_matrix))]
        return self.__graph
