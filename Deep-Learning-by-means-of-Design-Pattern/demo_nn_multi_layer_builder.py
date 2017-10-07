if __name__ == "__main__":
    from pydbm.nn.builders.nn_multi_layer_builder import NNMultiLayerBuilder
    from pydbm.activation.logistic_function import LogisticFunction
    from pydbm.nn.neural_network import NeuralNetwork
    import numpy as np
    import random
    import pandas as pd
    from pprint import pprint
    from sklearn.datasets import make_classification
    from sklearn.cross_validation import train_test_split

    data_tuple = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=10,
        n_classes=2,
        class_sep=1.0
    )
    data_tuple_x, data_tuple_y = data_tuple
    traning_x, test_x, traning_y, test_y = train_test_split(
        data_tuple_x,
        data_tuple_y,
        test_size=0.3,
        random_state=555
    )
    traning_y = traning_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)
    nn = NeuralNetwork(
        NNMultiLayerBuilder(),
        [traning_x.shape[1], 10, traning_y.shape[1]],
        [LogisticFunction(), LogisticFunction(), LogisticFunction()]
    )

    nn.learn(
        traning_x,
        traning_y,
        traning_count=1000,
        learning_rate=0.00001,
        momentum_factor=0.00001
    )
    for i in range(10):
        pred_arr = nn.predict(test_x[i])
        print(pred_arr)
