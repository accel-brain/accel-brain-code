#!/user/bin/env python
# -*- coding: utf-8 -*-
from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.activation.logistic_function import LogisticFunction
import numpy as np
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

    dbm = DeepBoltzmannMachine(
        DBMMultiLayerBuilder(),
        [traning_x.shape[1], 10, traning_x.shape[1]],
        LogisticFunction(),
        ContrastiveDivergence(),
        0.05
    )
    dbm.learn(traning_x, traning_count=1)
    print(dbm.get_feature_point_list(0))
