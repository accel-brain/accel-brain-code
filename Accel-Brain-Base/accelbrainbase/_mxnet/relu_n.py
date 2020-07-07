# -*- coding: utf-8 -*-
from mxnet.gluon.block import HybridBlock
from mxnet import gluon
import mxnet as mx


class ReLuN(HybridBlock):
    '''
    ReLu N(=6) layer.
    '''

    def __init__(self, min_n=0, max_n=6, **kwargs):
        '''
        Init.

        Args:
            min_n:      min of range.
            max_n:      max of range. If this value is `-1`, this class will use not ReLu6 but ReLu.

        '''
        super(ReLuN, self).__init__(**kwargs)
        self.__min_n = min_n
        self.__max_n = max_n

    def inference(self, observed_arr):
        '''
        Inference the labels.

        Args:
            observed_arr:   rank-2 Array like or sparse matrix as the observed data points.
                            The shape is: (batch size, feature points)

        Returns:
            `mxnet.ndarray` of inferenced feature points.
        '''
        return self(observed_arr)

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        if self.__max_n != -1:
            return F.clip(x, self.__min_n, self.__max_n)
        else:
            return F.Activation(x, "relu")
