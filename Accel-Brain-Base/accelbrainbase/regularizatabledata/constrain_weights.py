# -*- coding: utf-8 -*-
from accelbrainbase.regularizatable_data import RegularizatableData
import numpy as np


class ConstrainWeights(RegularizatableData):
    '''
    Regularization for weights matrix
    to repeat multiplying the weights matrix and `0.9`
    until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.

    References:
        - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.
    '''

    # Regularization for weights matrix
    # to repeat multiplying the weights matrix and `0.9`
    # until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
    __weight_limit = 0.9

    def __init__(self, weight_limit=0.9):
        '''
        Init.

        Args:
            weight_limit:       Regularization for weights matrix
                                to repeat multiplying the weights matrix and `0.9`
                                until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
        '''
        self.__weight_limit = weight_limit

    def regularize(self, params_dict):
        '''
        Regularize parameters.

        Args:
            params_dict:    is-a `mxnet.gluon.ParameterDict`.
        
        Returns:
            `mxnet.gluon.ParameterDict`
        '''
        for k, v in params_dict.items():
            if k[-6:] == "weight":
                params_dict[k] = self.constrain_weight(v)

        return params_dict

    def constrain_weight(self, weight_arr):
        square_weight_arr = weight_arr * weight_arr
        while np.nansum(square_weight_arr) > self.__weight_limit:
            weight_arr = weight_arr * 0.9
            square_weight_arr = weight_arr * weight_arr

        return weight_arr
