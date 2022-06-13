# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model import DiscriminativeModel

import numpy as np
from logging import getLogger

import torch


class EBDiscriminativeModel(DiscriminativeModel):
    '''
    Discriminative model, which discriminates true from fake,
    in the Energy-based Generative Adversarial Network(EBGAN).

    The Energy-based Generative Adversarial Network (EBGAN) model(Zhao, J., et al., 2016) which 
    views the discriminator as an energy function that attributes low energies to the regions 
    near the data manifold and higher energies to other regions. The Auto-Encoders have traditionally 
    been used to represent energy-based models. When trained with some regularization terms, 
    the Auto-Encoders have the ability to learn an energy manifold without supervision or negative examples. 
    This means that even when an energy-based Auto-Encoding model is trained to reconstruct a real sample, 
    the model contributes to discovering the data manifold by itself.

    References:
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def __init__(
        self, 
        model, 
        optimizer_f=None,
        learning_rate=1e-05,
        ctx="cpu", 
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        super().__init__(
            model=model, 
            optimizer_f=optimizer_f,
            learning_rate=learning_rate,
            ctx=ctx, 
            not_init_flag=not_init_flag,
        )

        logger = getLogger("accelbrainbase")
        self.__logger = logger

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
        Forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        inferenced_arr = self.model(x)
        inferenced_arr = inferenced_arr.reshape_as(x)
        mse_arr = torch.square(x - inferenced_arr)
        return torch.unsqueeze(
            torch.mean(
                mse_arr,
                dim=1,
            ),
            axis=-1
        )
