# -*- coding: utf-8 -*-
from pydbmmx.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from pydbmmx.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbmmx.approximation.contrastive_divergence import ContrastiveDivergence
from pydbmmx.activation.logistic_function import LogisticFunction
import numpy as np
import mxnet as mx
import random
import pandas as pd
from pprint import pprint
from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split


if __name__ == "__main__":
    '''
    '''

    data_tuple = make_classification(
        n_samples=20000,
        n_features=1000,
        n_informative=5,
        n_classes=5,
        class_sep=1.0,
        scale=0.1
    )
    data_tuple_x, data_tuple_y = data_tuple
    traning_x, test_x, traning_y, test_y = train_test_split(
        data_tuple_x,
        data_tuple_y,
        test_size=0.5,
        random_state=888
    )
    traning_x = mx.nd.array(traning_x)
    traning_y = mx.nd.array(traning_y)
    test_x = mx.nd.array(test_x)
    test_y = mx.nd.array(test_y)

    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [traning_x.shape[1], 10, traning_x.shape[1]],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    row = traning_x.shape[0]
    feature_points_list = [None] * row
    traning_count = 10
    for t in range(traning_count):
        for i in range(row):
            data_arr = traning_x[i]
            dbm.learn(
                observed_data_arr=data_arr,
                traning_count=1
            )
            if t == traning_count - 1:
                feature_points_arr = dbm.get_feature_point()
                feature_points_list[i] = feature_points_arr.asnumpy().tolist()

    feature_points_arr = mx.nd.array(feature_points_list)
    print(feature_points_arr)
