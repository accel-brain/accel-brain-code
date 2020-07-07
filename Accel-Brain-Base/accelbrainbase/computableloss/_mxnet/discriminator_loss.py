# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss
from mxnet.gluon.loss import Loss
import numpy as np
from mxnet.gluon.loss import _reshape_like, _apply_weighting


class DiscriminatorLoss(Loss, ComputableLoss):
    '''
    Loss function of discriminators in Generative Adversarial Networks(GANs).

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    def __init__(
        self, 
        weight=None,
        batch_axis=0, 
        **kwargs
    ):
        '''
        Init.

        Args:
            weight:                 Global scalar weight for loss.
            **kwargs:               Any parmaeter.
        '''
        super(DiscriminatorLoss, self).__init__(
            weight, 
            batch_axis, 
            **kwargs
        )
        self._weight = weight

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
        loss = F.log(true_posterior_arr + 1e-08) + F.log(1 - generated_posterior_arr + 1e-08)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)
