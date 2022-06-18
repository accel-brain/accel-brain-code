# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model import DiscriminativeModel
import numpy as np
from logging import getLogger
import torch
from torch import nn


class MultimodalDiscriminativeModel(DiscriminativeModel):
    '''
    Discriminative model, which discriminates multi-modal observed data points.
    '''

    def __init__(
        self, 
        model_list, 
        final_model,
        learning_rate=1e-05,
        ctx="cpu", 
    ):
        '''
        Init.

        Args:
            model_list:                     `list` of `mxnet.gluon.hybrid.hybridblock.HybridBlock`s.
            final_model:                    is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True
        logger = getLogger("accelbrainbase")
        self.__logger = logger

        super().__init__(
            model=final_model,
            learning_rate=learning_rate,
            ctx=ctx, 
        )
        self.model = final_model
        self.model_list = model_list

        self.init_deferred_flag = init_deferred_flag

    def inference(self, observed_arr):
        '''
        Draw samples from the fake distribution.

        Args:
            observed_arr:       `mxnet.ndarray` or `mxnet.symbol` of observed data points.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        return self.forward(observed_arr)

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        inferenced_arr = self.model(x)
        return inferenced_arr
