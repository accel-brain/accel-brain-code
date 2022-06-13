# -*- coding: utf-8 -*-
import numpy as np
from accelbrainbase.computableloss._torch.discriminator_loss import DiscriminatorLoss
import torch
import torch.nn as nn


class EBDiscriminatorLoss(DiscriminatorLoss):
    '''
    Loss function of discriminators in Generative Adversarial Networks(GANs).
    '''

    def __init__(
        self, 
        weight=None,
        margin=1.0,
        margin_decay_rate=0.1,
        margin_decay_epoch=50,
    ):
        '''
        Init.

        Args:
            weight:                 Global scalar weight for loss.
            **kwargs:               Any parmaeter.
        '''
        super(EBDiscriminatorLoss, self).__init__()
        self.__weight = weight
        self.__margin = margin
        self.__margin_decay_rate = margin_decay_rate
        self.__margin_decay_epoch = margin_decay_epoch

        self.epoch = 0

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
        zero_arr = torch.Tensor([0])
        zero_arr = zero_arr.to(true_posterior_arr.device)

        loss_arr = true_posterior_arr + torch.maximum(
            zero_arr, 
            self.__margin - generated_posterior_arr
        )
        loss_arr = loss_arr * self.__weight

        self.epoch += 1
        if self.epoch % self.__margin_decay_epoch == 0:
            self.__margin = self.__margin * self.__margin_decay_rate

        return torch.mean(loss_arr)
