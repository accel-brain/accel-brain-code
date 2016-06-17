#!/user/bin/env python
# -*- coding: utf-8 -*-
from deeplearning.synapse_list import Synapse
from deeplearning.activation.sigmoid_function import SigmoidFunction


class NeuralNetworkGraph(Synapse):
    '''
    バックプロパゲーションを実行することで重みを更新する
    ニューラルネットワーク
    '''

    # 出力層のフラグ
    __output_layer_flag = False
    # シグモイド関数
    __sigmoid_function = None
    # 運動量係数の辞書
    __momentum_factor_dict = {}

    def __init__(self, output_layer_flag=False):
        '''
        初期化
        '''
        if isinstance(output_layer_flag, bool) is False:
            raise TypeError()
        self.__output_layer_flag = output_layer_flag

        self.__sigmoid_function = SigmoidFunction()

    def back_propagate(self, propagated_list, learning_rate=0.05, momentum_factor=0.1):
        '''
        「中間層における最上位層」からそれ以降の中間層、
        及び入力層までの組み合わせを前提として、
        フォワードプロパゲーションを実行する

        Args:
            propagated_list:    伝播内容のリスト、出力層ならば目的変数のリストになる
            learning_rate:      学習率
            momentum_factor:    運動量係数

        Returns:
            伝播内容のリスト
        '''
        diff_list = []
        if self.__output_layer_flag is False:
            '''
            「中間層における最上位層」からそれ以降の中間層、
            及び入力層までの組み合わせ
            '''
            if len(self.deeper_neuron_list) != len(propagated_list):
                print(len(self.deeper_neuron_list))
                print(len(propagated_list))
                raise ValueError()

            for i in range(len(self.shallower_neuron_list)):
                error = 0.0
                for j in range(len(self.deeper_neuron_list)):
                    self.diff_weights_dict.setdefault((i, j), 0.0)
                    error += propagated_list[j] * self.diff_weights_dict[(i, j)]

                activity = self.shallower_neuron_list[i].activity
                diff_list.append(
                    self.__sigmoid_function.derivative(activity) * error
                )
        else:
            '''
            出力層から「中間層における最上位層」までの組み合わせ
            '''
            diff_list = []
            for i in range(len(self.deeper_neuron_list)):
                error = propagated_list[i] - self.deeper_neuron_list[i].activity

                activity = self.deeper_neuron_list[i].activity
                diff_list.append(
                    self.__sigmoid_function.derivative(activity) * error
                )

        for i in range(len(self.shallower_neuron_list)):
            for j in range(len(self.deeper_neuron_list)):
                self.diff_weights_dict.setdefault((i, j), 0.0)
                self.__momentum_factor_dict.setdefault((i, j), 0.0)
                diff = diff_list[j] * self.shallower_neuron_list[i].activity
                self.diff_weights_dict[(i, j)] += learning_rate * diff + momentum_factor * self.__momentum_factor_dict[(i, j)]
                self.__momentum_factor_dict[(i, j)] = diff

        # 重みの更新
        self.learn_weights()
        # バイアスの更新
        [neuron.update_bias(learning_rate) for neuron in self.shallower_neuron_list]
        [neuron.update_bias(learning_rate) for neuron in self.deeper_neuron_list]

        return diff_list
