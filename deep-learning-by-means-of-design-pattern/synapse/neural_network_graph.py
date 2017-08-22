#!/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from deeplearning.synapse_list import Synapse
from deeplearning.activation.logistic_function import LogisticFunction


class NeuralNetworkGraph(Synapse):
    '''
    バックプロパゲーションを実行することで重みを更新する
    ニューラルネットワーク
    '''

    # 出力層のフラグ
    __output_layer_flag = False
    # ロジスティック関数
    __logistic_function = None
    # 運動量係数の辞書
    __momentum_factor_arr = None

    def __init__(self, output_layer_flag=False):
        '''
        初期化
        '''
        if isinstance(output_layer_flag, bool) is False:
            raise TypeError()
        self.__output_layer_flag = output_layer_flag
        self.__logistic_function = LogisticFunction()

    def back_propagate(
        self,
        propagated_list,
        learning_rate=0.05,
        momentum_factor=0.1,
        back_nn_list=None,
        back_nn_index=0
    ):
        '''
        再帰的に自己自身を呼び出すことで
        各層間のバックプロパゲーションを順次実行していく

        Args:
            propagated_list:    伝播内容のリスト、出力層ならば目的変数のリストになる
            learning_rate:      学習率
            momentum_factor:    運動量係数
            back_nn_list:       再帰的に呼び出すニューラルネットワークのオブジェクト
            back_nn_index:      再帰中に呼び出しているオブジェクトの添え字番号

        Returns:
            伝播内容と活性度のtupleのリスト
        '''
        if self.__output_layer_flag is True:
            '''
            出力層から「中間層における最上位層」までの組み合わせ
            一旦外部から入力された教師データと最上位である出力層の出力値との誤差を検出した後に、
            それを下層となる中間層に伝播する
            '''
            if len(self.deeper_neuron_list) != len(propagated_list):
                raise IndexError()

            diff_list = [self.deeper_neuron_list[j].activity - propagated_list[j] for j in range(len(self.deeper_neuron_list))]
        else:
            diff_list = propagated_list

        diff_list = list(np.nan_to_num(np.array(diff_list)))

        diff_arr = np.array([[diff_list[k]] * len(self.shallower_neuron_list) for k in range(len(diff_list))]).T
        if self.__momentum_factor_arr is not None:
            momentum_arr = self.__momentum_factor_arr * momentum_factor
        else:
            momentum_arr = np.ones(diff_arr.shape) * momentum_factor

        self.diff_weights_arr = (learning_rate * diff_arr) + momentum_arr
        self.__momentum_factor_arr = diff_arr
        self.weights_arr = np.nan_to_num(self.weights_arr)
        error_arr = diff_arr * self.weights_arr
        error_list = error_arr.sum(axis=1)
        back_propagated_list = [self.__logistic_function.derivative(self.shallower_neuron_list[i].activity) * error_list[i] for i in range(len(self.shallower_neuron_list))]

        # 規格化
        if len(back_propagated_list) > 1 and sum(back_propagated_list) != 0:
            back_propagated_arr = np.array(back_propagated_list)
            back_propagated_arr = back_propagated_arr / back_propagated_arr.sum()
            back_propagated_arr = np.nan_to_num(back_propagated_arr)
            back_propagated_list = list(back_propagated_arr)

        # 重みの更新
        self.learn_weights()

        # バイアスの更新
        [neuron.update_bias(learning_rate) for neuron in self.shallower_neuron_list]
        [neuron.update_bias(learning_rate) for neuron in self.deeper_neuron_list]

        if back_nn_list is not None:
            if back_nn_index < len(back_nn_list) - 1:
                back_nn_list[back_nn_index + 1].back_propagate(
                    propagated_list=back_propagated_list,
                    learning_rate=learning_rate,
                    momentum_factor=momentum_factor,
                    back_nn_list=back_nn_list,
                    back_nn_index=back_nn_index + 1
                )
