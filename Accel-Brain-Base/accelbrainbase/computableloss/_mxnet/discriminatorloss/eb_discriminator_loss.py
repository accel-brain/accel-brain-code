# -*- coding: utf-8 -*-
from mxnet.gluon.loss import Loss
import numpy as np
from mxnet.gluon.loss import _reshape_like, _apply_weighting
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss


class EBDiscriminatorLoss(DiscriminatorLoss):
    '''
    Loss function of discriminators in Generative Adversarial Networks(GANs).
    '''

    def __init__(
        self, 
        weight=None,
        batch_axis=0, 
        margin=1.0,
        margin_decay_rate=0.1,
        margin_decay_epoch=50,
        **kwargs
    ):
        '''
        Init.

        Args:
            weight:                 Global scalar weight for loss.
            **kwargs:               Any parmaeter.
        '''
        super(EBDiscriminatorLoss, self).__init__(
            weight, 
            batch_axis, 
            **kwargs
        )
        self._weight = weight
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
        return self(true_posterior_arr, generated_posterior_arr)

    def hybrid_forward(
        self, 
        F, 
        true_posterior_arr,
        generated_posterior_arr,
        sample_weight=None
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
        loss = true_posterior_arr + F.maximum(0, self.__margin - generated_posterior_arr)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)

        self.epoch += 1
        if self.epoch % self.__margin_decay_epoch == 0:
            self.__margin = self.__margin * self.__margin_decay_rate

        return F.mean(loss, axis=self._batch_axis, exclude=True)
