#!/user/bin/env python
# -*- coding: utf-8 -*-
import sys

from deeplearning.approximation.interface.approximate_interface import ApproximateInterface


class ContrastiveDivergence(ApproximateInterface):
    '''
    Contrastive Divergence法

    概念的にはpositive phaseとnegative phaseの区別を
    wakeとsleepの区別に対応付けて各メソッドを命名している。

    @TODO(chimera0):numpyのベクトル計算関連の関数を再利用することで計算を高速化させる
    '''

    # ニューロンのグラフ
    __graph = None
    # 学習率
    __learning_rate = 0.5

    def approximate_learn(self, graph, learning_rate, observed_data_matrix, traning_count=1000):
        '''
        インターフェイスの実現
        近似による学習

        Args:
            graph:                ニューロンのグラフ
            learning_rate:        学習率
            observed_data_matrix: 観測データ点
            traning_count:        訓練回数

        '''
        self.__graph = graph
        self.__learning_rate = learning_rate
        [self.__wake_sleep_learn(observed_data_matrix) for i in range(traning_count)]
        return self.__graph

    def __wake_sleep_learn(self, observed_data_matrix):
        '''
        目覚めと眠りのフェーズを逐次実行して学習する

        Args:
            observed_data_matrix:   観測データ点リスト
        '''
        for observed_data_list in observed_data_matrix:
            self.wake(observed_data_list)
            self.sleep()
            self.learn()

    def wake(self, observed_data_list):
        '''
        目覚めた状態で外部環境を観測する

        Args:
            observed_data_matrix:   観測データ点
        '''
        if len(observed_data_list) != len(self.__graph.visible_neuron_list):
            print(len(observed_data_list))
            print(len(self.__graph.visible_neuron_list))
            raise ValueError("len(observed_data_list) != len(self.__graph.links_weights)")

        # 観測データ点を可視ニューロンに入力する
        [self.__graph.visible_neuron_list[i].observe_data_point(observed_data_list[i]) for i in range(len(observed_data_list))]
        # 隠れ層のニューロンの発火状態を更新する
        for i in range(len(self.__graph.visible_neuron_list)):
            [self.__graph.hidden_neuron_list[j].hidden_update_state(self.__graph.weights_dict[i, j], self.__graph.visible_neuron_list[i].activity) for j in range(len(self.__graph.hidden_neuron_list))]
        # ヘブ規則によりリンクの重みを更新する
        self.__graph.update(self.__learning_rate)
        # 可視層のバイアスを更新する
        [self.__graph.visible_neuron_list[i].update_bias(self.__learning_rate) for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスを更新する
        [self.__graph.hidden_neuron_list[i].update_bias(self.__learning_rate) for i in range(len(self.__graph.hidden_neuron_list))]

    def sleep(self):
        '''
        夢を見る
        自由連想
        '''
        # 可視層のニューロンの発火状態を更新する
        for j in range(len(self.__graph.hidden_neuron_list)):
            [self.__graph.visible_neuron_list[i].visible_update_state(self.__graph.weights_dict[(i, j)], self.__graph.hidden_neuron_list[j].activity) for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のニューロンの発火状態を更新する
        for i in range(len(self.__graph.visible_neuron_list)):
            [self.__graph.hidden_neuron_list[j].hidden_update_state(self.__graph.weights_dict[(i, j)], self.__graph.visible_neuron_list[i].activity) for j in range(len(self.__graph.hidden_neuron_list))]
        # ヘブ規則によりリンクの重みを逆更新する
        self.__graph.update((-1) * self.__learning_rate)
        # 可視層のバイアスを更新する
        [self.__graph.visible_neuron_list[i].update_bias((-1) * self.__learning_rate) for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスを更新する
        [self.__graph.hidden_neuron_list[i].update_bias((-1) * self.__learning_rate) for i in range(len(self.__graph.hidden_neuron_list))]

    def learn(self):
        '''
        バイアスと重みを学習する
        '''
        # 可視層のバイアスの学習
        [self.__graph.visible_neuron_list[i].learn_bias() for i in range(len(self.__graph.visible_neuron_list))]
        # 隠れ層のバイアスの学習
        [self.__graph.hidden_neuron_list[i].learn_bias() for i in range(len(self.__graph.hidden_neuron_list))]
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
        # 観測データ点を可視ニューロンに入力する
        for observed_data_list in observed_data_matrix:
            [self.__graph.visible_neuron_list[i].observe_data_point(observed_data_list[i]) for i in range(len(observed_data_list))]
            # 可視層のニューロンの発火状態を更新する
            for j in range(len(self.__graph.hidden_neuron_list)):
                [self.__graph.visible_neuron_list[i].visible_update_state(self.__graph.weights_dict[(i, j)], self.__graph.hidden_neuron_list[j].activity) for i in range(len(self.__graph.visible_neuron_list))]
            # 隠れ層のニューロンの発火状態を更新する
            for i in range(len(self.__graph.visible_neuron_list)):
                [self.__graph.hidden_neuron_list[j].hidden_update_state(self.__graph.weights_dict[(i, j)], self.__graph.visible_neuron_list[i].activity) for j in range(len(self.__graph.hidden_neuron_list))]

        return self.__graph
