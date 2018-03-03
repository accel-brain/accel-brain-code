# -*- coding: utf-8 -*-
from pydbm.dbm.builders.rt_rbm_simple_builder import RTRBMSimpleBuilder
from pydbm.dbm.restricted_boltzmann_machines import RestrictedBoltzmannMachine
from pydbm.approximation.rt_rbm_cd import RTRBMCD
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.softmax_function import SoftmaxFunction
import numpy as np
import random
import pandas as pd
from pprint import pprint

np.seterr(all='raise')

if __name__ == "__main__":
    '''
    '''
    arr = np.arange(333, dtype=np.float64)
    arr = np.c_[
        arr+0,
        arr+1,
        arr+2,
        arr+3,
        arr+4,
        arr+5,
        arr+6,
        arr+7
    ]
    arr = np.array(arr.tolist() * 100)
    arr = arr / 333

    print(arr.shape)

    rtrbm_builder = RTRBMSimpleBuilder()
    rtrbm_builder.learning_rate = 0.00001
    rtrbm_builder.visible_neuron_part(LogisticFunction(), arr.shape[1])
    rtrbm_builder.hidden_neuron_part(LogisticFunction(), 3)
    rtrbm_builder.rnn_neuron_part(LogisticFunction())
    rtrbm_builder.graph_part(RTRBMCD())
    rbm = rtrbm_builder.get_result()

    for i in range(arr.shape[0]):
        rbm.approximate_learning(
            arr[i],
            traning_count=1, 
            batch_size=200
        )

    test_arr = arr[0]
    result_list = [None] * arr.shape[0]
    for i in range(arr.shape[0]):
        rbm.approximate_inferencing(
            test_arr,
            traning_count=1, 
            r_batch_size=-1
        )
        result_list[i] = test_arr = rbm.graph.visible_activity_arr
    print(np.array(result_list))
