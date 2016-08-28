#!/user/bin/env python
# -*- coding: utf-8 -*-
from deeplearning.synapse_list import Synapse
from deeplearning.activation.logistic_function import LogisticFunction


class NeuralNetworkGraph(Synapse):
    '''
    バックプロパゲーションを実行することで重みを更新する
    ニューラルネットワーク
    '''

    # 出力層のフラグ
    __output_layer_flag = False
    # シグモイド関数
    __logistic_function = None
    # 運動量係数の辞書
    __momentum_factor_dict = {}

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
                print(len(self.deeper_neuron_list))
                print(len(propagated_list))
                print(propagated_list)
                raise IndexError()

            diff_list = []
            for j in range(len(self.deeper_neuron_list)):
                # 教師データリストの諸要素と出力層の各ニューロンの多重度は1:1
                error = propagated_list[j] - self.deeper_neuron_list[j].activity
                activity = self.deeper_neuron_list[j].activity
                # 外部から出力層に入力された場合、
                # 言うなれば伝播すべき誤差データが無いために、ここで生成する
                diff_list.append(
                    self.__logistic_function.derivative(activity) * error
                )
        else:
            diff_list = propagated_list

        back_propagated_list = []
        for i in range(len(self.shallower_neuron_list)):
            activity = self.shallower_neuron_list[i].activity
            error = 0.0
            for j in range(len(self.deeper_neuron_list)):
                self.diff_weights_dict.setdefault((i, j), 0.0)
                self.__momentum_factor_dict.setdefault((i, j), 0.0)

                diff = diff_list[j] * activity
                momentum = momentum_factor * self.__momentum_factor_dict[(i, j)]
                self.diff_weights_dict[(i, j)] += (learning_rate * diff) + momentum
                self.__momentum_factor_dict[(i, j)] = diff

                error += diff_list[j] * self.weights_dict[(i, j)]

            back_propagated_list.append(
                self.__logistic_function.derivative(activity) * error
            )

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
