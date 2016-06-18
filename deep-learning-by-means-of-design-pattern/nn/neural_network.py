#!/user/bin/env python
# -*- coding: utf-8 -*-
from multipledispatch import dispatch
from deeplearning.nn.nn_director import NNDirector
from deeplearning.nn.interface.nn_builder import NNBuilder


class NeuralNetwork(object):
    '''
    ニューラルネットワークのオブジェクト
    '''

    # ニューラルネットワークのグラフ
    __nn_list = []

    # 評価時に参照する（ハイパー）パラメタの記録用辞書
    __hyper_param_dict = {}

    @dispatch(NNBuilder, int, int, int, list)
    def __init__(
        self,
        nn_builder,
        input_neuron_count,
        hidden_neuron_count,
        output_neuron_count,
        activating_function_list
    ):
        '''
        ニューラルネットワークを初期化する

        Args:
            input_neuron_count:         入力層ニューロン数
            hidden_neuron_count:   　    隠れ層のニューロン数
            output_neuron_count:        出力層ニューロン数
            activating_function_list:   活性化関数のリスト 入力層、隠れ層、出力層の順
        '''
        nn_director = NNDirector(
            nn_builder=nn_builder
        )
        nn_director.nn_construct(
            neuron_assign_list=[input_neuron_count, hidden_neuron_count, output_neuron_count],
            activating_function_list=activating_function_list
        )

        self.__nn_list = nn_director.nn_list

        self.__hyper_param_dict = {
            "neuron_assign_list": [input_neuron_count, hidden_neuron_count, output_neuron_count],
            "activating_function_list": [type(a) for a in activating_function_list]
        }

    @dispatch(NNBuilder, list, list)
    def __init__(
        self,
        nn_builder,
        neuron_assign_list,
        activating_function_list
    ):
        '''
        ニューラルネットワークを初期化する

        Args:
            neuron_assign_list:         各層のニューロンの個数
                                        0番目が入力層で、最後が出力層で、
                                        それ以外の値が隠れ層に対応する
            activating_function_list:   活性化関数のリスト
                                        引数：neuron_assign_listの構成に合わせる
        '''
        nn_director = NNDirector(
            nn_builder=nn_builder
        )
        nn_director.nn_construct(
            neuron_assign_list=neuron_assign_list,
            activating_function_list=activating_function_list
        )

        self.__nn_list = nn_director.nn_list

        self.__hyper_param_dict = {
            "neuron_assign_list": neuron_assign_list,
            "activating_function_list": [type(a) for a in activating_function_list]
        }

    def learn(
        self,
        traning_data_matrix,
        class_data_matrix,
        learning_rate=0.5,
        momentum_factor=0.1,
        traning_count=1000
    ):
        '''
        フォワードプロパゲーションとバックプロパゲーションを交互に実行し続ける

        Args:
            traning_data_matrix:    訓練データ
            class_data_matrix:      教師データ
            learning_rate:          学習率
            momentum_factor:        運動量係数
            traning_count:          訓練回数

        '''
        if len(traning_data_matrix) != len(class_data_matrix):
            raise ValueError()

        for i in range(traning_count):
            for j in range(len(traning_data_matrix)):
                self.forward_propagate(traning_data_matrix[j])
                self.back_propagate(
                    test_data_list=class_data_matrix[j],
                    learning_rate=learning_rate,
                    momentum_factor=momentum_factor
                )

        self.__hyper_param_dict["traning_count"] = traning_count
        self.__hyper_param_dict["momentum_factor"] = momentum_factor
        self.__hyper_param_dict["learning_rate"] = learning_rate

    def forward_propagate(self, input_data_list):
        '''
        フォワードプロパゲーションを実行する

        Args:
            input_data_list:  訓練データ

        '''
        nn_from_input_to_hidden_layer = self.__nn_list[0]
        nn_hidden_layer_list = []
        for i in range(1, len(self.__nn_list) - 1):
            nn_hidden_layer_list.append(self.__nn_list[i])
        nn_to_output_layer = self.__nn_list[-1]

        for i in range(len(input_data_list)):
            nn_from_input_to_hidden_layer.shallower_neuron_list[i].observe_data_point(input_data_list[i])

        for j in range(len(nn_from_input_to_hidden_layer.deeper_neuron_list)):
            link_value = 0.0
            for i in range(len(nn_from_input_to_hidden_layer.shallower_neuron_list)):
                activity = nn_from_input_to_hidden_layer.shallower_neuron_list[i].activity
                weight = nn_from_input_to_hidden_layer.weights_dict[(i, j)]
                link_value += activity * weight
            nn_from_input_to_hidden_layer.deeper_neuron_list[j].hidden_update_state(link_value)

        for nn_hidden_layer in nn_hidden_layer_list:
            for j in range(len(nn_hidden_layer.deeper_neuron_list)):
                link_value = 0.0
                for i in range(len(nn_hidden_layer.shallower_neuron_list)):
                    nn_hidden_layer.diff_weights_dict.setdefault((i, j), 0.0)
                    activity = nn_hidden_layer.shallower_neuron_list[i].activity
                    weight = nn_hidden_layer.diff_weights_dict[(i, j)]
                    link_value += activity * weight
                nn_hidden_layer.deeper_neuron_list[j].hidden_update_state(link_value)

        for j in range(len(nn_to_output_layer.deeper_neuron_list)):
            link_value = 0.0
            for i in range(len(nn_to_output_layer.shallower_neuron_list)):
                activity = nn_to_output_layer.shallower_neuron_list[i].activity
                weight = nn_to_output_layer.weights_dict[(i, j)]
                link_value += activity * weight
            nn_to_output_layer.deeper_neuron_list[j].output_update_state(link_value)

    def back_propagate(self, test_data_list, learning_rate=0.05, momentum_factor=0.1):
        '''
        バックプロパゲーションを実行する

        Args:
            test_data_list:     検証用データ
            learning_rate:      学習率
            momentum_factor:    運動量係数
        '''
        back_nn_list = [back_nn for back_nn in reversed(self.__nn_list)]
        back_nn_list[0].back_propagate(
            propagated_list=test_data_list,
            learning_rate=learning_rate,
            momentum_factor=momentum_factor,
            back_nn_list=back_nn_list,
            back_nn_index=0
        )

    def predict(self, test_data_list):
        '''
        予測する

        Args:
            test_data_matrix:   検証用データ

        Returns:
            予測結果となる出力値
        '''

        output_data_list = []
        self.forward_propagate(test_data_list)
        for output_neuron in self.__nn_list[-1].deeper_neuron_list:
            output_data_list.append(output_neuron.release())

        return output_data_list

    def evaluate_bool(self, test_data_matrix, class_data_matrix):
        '''
        教師データの訓練後の予測を実行して、モデル性能をF値などの指標で算出する
        目的変数が二値(0/1)の場合の簡易版

        Args:
            test_data_matrix:   テストデータ

        Returns:
            次の辞書を返す
            {
                "tp":                 True Positive,
                "fp":                 False Positive,
                "tn":                 True Negative,
                "fn":                 False Negative,
                "precision":          適合率,
                "recall":             再現率,
                "f":                  F値,
                "<<（ハイパー）パラメタ名>>": <<値>>
            }
        '''
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(test_data_matrix)):
            test_data_list = test_data_matrix[i]
            class_data_list = class_data_matrix[i]

            binary_result = self.predict(test_data_list)[0]

            if class_data_list[0] == binary_result and binary_result == 0:
                tp += 1
            elif class_data_list[0] == binary_result and binary_result == 1:
                tn += 1
            elif class_data_list[0] != binary_result and binary_result == 0:
                fp += 1
            else:
                fn += 1

        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0.0
        try:
            f = 2 * ((precision * recall) / (precision + recall))
        except ZeroDivisionError:
            f = 0.0

        result_dict = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f": f
        }
        [result_dict.setdefault(key, val) for key, val in self.__hyper_param_dict.items()]

        return result_dict
