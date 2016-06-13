#!/user/bin/env python
# -*- coding: utf-8 -*-

from deeplearning.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from deeplearning.dbm.builders.dbm_3_layer_builder import DBM3LayerBuilder
from deeplearning.approximation.contrastive_divergence import ContrastiveDivergence
from deeplearning.activation.sigmoid_function import SigmoidFunction
from sklearn.datasets import load_iris
import numpy as np
import random
import pandas as pd
from pprint import pprint


if __name__ == "__main__":

    # irisのデータセットを参照する
    data_arr = load_iris()
    feature_arr = data_arr["data"]
    target_arr = data_arr["target"]

    target_arr = target_arr.reshape((len(target_arr), 1))
    data_arr = np.hstack((target_arr, feature_arr))

    # 目的変数を0-1に限定する
    data_matrix = list(data_arr[:99])

    # シャッフルし、訓練用データとテスト用データに分割する
    random.shuffle(data_matrix)
    traning_data_matrix = data_matrix[:69]
    test_data_matrix = data_matrix[70:99]

    evaluate_data_list = []

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=2)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )

    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=100)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)
    evaluate_data = pd.DataFrame(evaluate_data_list)
    evaluate_data = evaluate_data.sort_values(by=["f", "precision", "recall"], ascending=False)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )

    dbm.learn(traning_data_matrix, traning_count=1000)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_data_matrix, traning_count=1000)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        10,
        5,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=1000)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)

    dbm = DeepBoltzmannMachine(
        DBM3LayerBuilder(),
        len(data_matrix[0]),
        3,
        2,
        SigmoidFunction(),
        ContrastiveDivergence(),
        0.5
    )
    dbm.learn(traning_data_matrix, traning_count=1000)
    evaluate_dict = dbm.evaluate_bool(test_data_matrix)
    evaluate_data_list.append(evaluate_dict)
    evaluate_data = pd.DataFrame(evaluate_data_list)
    evaluate_data = evaluate_data.sort_values(by=["f", "precision", "recall"], ascending=False)

    data_columns = [
        "visible_neuron_count",
        "feature_neuron_count",
        "hidden_neuron_count",
        "learning_rate",
        "traning_count",
        "precision",
        "recall",
        "f"
    ]

    print(evaluate_data[data_columns])
    '''
    test    
    '''
