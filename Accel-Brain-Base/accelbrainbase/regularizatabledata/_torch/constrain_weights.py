# -*- coding: utf-8 -*-
from accelbrainbase.regularizatabledata.constrain_weights import ConstrainWeights as _ConstrainWeights
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainWeights(_ConstrainWeights):
    '''
    Regularization for weights matrix
    to repeat multiplying the weights matrix and `0.9`
    until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.

    References:
        - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The Journal of Machine Learning Research, 15(1), 1929-1958.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.
    '''

    def constrain_weight(self, weight_arr):
        square_weight_arr = weight_arr * weight_arr
        while torch.nansum(square_weight_arr) > self.weight_limit:
            weight_arr = weight_arr * 0.9
            square_weight_arr = weight_arr * weight_arr

        return weight_arr
