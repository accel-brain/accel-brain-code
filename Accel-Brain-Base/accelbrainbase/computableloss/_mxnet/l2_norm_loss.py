# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss
from mxnet.gluon.loss import Loss
from mxnet.gluon.loss import _reshape_like, _apply_weighting


class L2NormLoss(Loss, ComputableLoss):
    '''
    Loss function that computes L2 norm.
    '''

    def __init__(self, weight=1.0, batch_axis=0, **kwargs):
        '''
        Init.

        Args:
            weight:         Global scalar weight for loss.
            batch_axis:     The axisthat represents mini-batch.
            **kwargs:       Any parmaeter.
        '''
        super(L2NormLoss, self).__init__(weight, batch_axis, **kwargs)

    def compute(self, pred_arr, real_arr):
        '''
        Compute loss.

        Args:
            pred_arr:       Inferenced results.
            real_arr:       Real results.
        
        Returns:
            Tensor of losses.
        '''
        return self(real_arr, pred_arr)

    def hybrid_forward(self, F, orign_arr, dest_arr, sample_weight=None):
        '''
        Forward propagation, computing L2 norm.

        Args:
            F:           `mxnet.ndarray` or `mxnet.symbol`.
            orign_arr:   `mxnet.ndarray` or `mxnet.symbol` of origins.
            dest_arr:    `mxnet.ndarray` or `mxnet.symbol` of destinations.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        dest_arr = _reshape_like(F, dest_arr, orign_arr)
        loss = F.sqrt(F.mean(F.square(orign_arr - dest_arr), axis=1))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(
            loss,
            axis=self._batch_axis,
            exclude=True
        )
