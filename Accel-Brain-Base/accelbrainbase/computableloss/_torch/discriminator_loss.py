# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
import torch
import torch.nn as nn


class DiscriminatorLoss(nn.modules.loss._Loss, ComputableLoss):
    '''
    Loss function of discriminators in Generative Adversarial Networks(GANs).

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
        super(DiscriminatorLoss, self).__init__()
        self.__weight = weight

    def compute(self, true_posterior_arr, generated_posterior_arr):
        '''
        Compute loss.

        Args:
            true_posterior_arr:            Real samples.
            generated_posterior_arr:       Generated samples.

        Returns:
            Tensor of losses.
        '''
        return self.forward(true_posterior_arr, generated_posterior_arr)

    def forward(
        self, 
        true_posterior_arr,
        generated_posterior_arr,
    ):
        '''
        Forward propagation, computing losses.

        Args:
            F:                          `mxnet.ndarray` or `mxnet.symbol`.
            true_posterior_arr:         `mxnet.ndarray` or `mxnet.symbol` of true posterior
                                        inferenced by the discriminator.

            generated_posterior_arr:    `mxnet.ndarray` or `mxnet.symbol` of fake posterior
                                        inferenced by the generator.

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        loss_arr = torch.log(true_posterior_arr + 1e-08) + torch.log(1 - generated_posterior_arr + 1e-08)
        loss_arr = loss_arr * self.__weight
        return torch.mean(loss_arr)
