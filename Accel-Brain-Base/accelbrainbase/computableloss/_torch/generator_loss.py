# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss

import numpy as np
import torch
import torch.nn as nn


class GeneratorLoss(nn.modules.loss._Loss, ComputableLoss):
    '''
    Loss function of generators in Generative Adversarial Networks(GANs).

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    def __init__(
        self, 
        weight=None,
    ):
        '''
        Init.

        Args:
            weight:                 Global scalar weight for loss.
            **kwargs:               Any parmaeter.
        '''
        super(GeneratorLoss, self).__init__()
        self.__weight = weight

    def compute(self, generated_posterior_arr):
        '''
        Compute loss.

        Args:
            generated_posterior_arr:       Generated samples.

        Returns:
            Tensor of losses.
        '''
        return self.forward(generated_posterior_arr)

    def forward(
        self, 
        generated_posterior_arr,
    ):
        '''
        Forward propagation, computing losses.

        Args:
            F:                          `mxnet.ndarray` or `mxnet.symbol`.
            generated_posterior_arr:    `mxnet.ndarray` or `mxnet.symbol` of fake posterior
                                        inferenced by the generator.

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        loss_arr = torch.log(1 - generated_posterior_arr + 1e-08)
        loss_arr = loss_arr * self.__weight
        return torch.mean(loss_arr)
