# -*- coding: utf-8 -*-
from mxnet.gluon.loss import Loss
from accelbrainbase.computable_loss import ComputableLoss
import numpy as np
from mxnet.gluon.loss import _reshape_like, _apply_weighting


class DRCNLoss(Loss, ComputableLoss):
    '''
    Loss function of Deep Reconstruction-Classification Networks.

    Deep Reconstruction-Classification Network(DRCN) is a convolutional network 
    that jointly learns two tasks: 
    
    1. supervised source label prediction.
    2. unsupervised target data reconstruction. 

    Ideally, a discriminative representation should model both the label and 
    the structure of the data. Based on that intuition, Ghifary, M., et al.(2016) hypothesize 
    that a domain-adaptive representation should satisfy two criteria:
    
    1. classify well the source domain labeled data.
    2. reconstruct well the target domain unlabeled data, which can be viewed as an approximate of the ideal discriminative representation.

    The encoding parameters of the DRCN are shared across both tasks, 
    while the decoding parameters are sepa-rated. The aim is that the learned label 
    prediction function can perform well onclassifying images in the target domain
    thus the data reconstruction can beviewed as an auxiliary task to support the 
    adaptation of the label prediction.

    References:
        - Ghifary, M., Kleijn, W. B., Zhang, M., Balduzzi, D., & Li, W. (2016, October). Deep reconstruction-classification networks for unsupervised domain adaptation. In European Conference on Computer Vision (pp. 597-613). Springer, Cham.
    '''

    def __init__(
        self, 
        axis=-1,
        sparse_label=False,
        rc_lambda=0.75,
        from_logits=False,
        log_softmax_flag=True,
        weight=1.0,
        classification_weight=None, 
        reconstruction_weight=None, 
        grad_clip_threshold=0.0,
        batch_axis=0, 
        **kwargs
    ):
        '''
        Init.

        Args:
            axis:                   The axis to sum over when computing softmax and entropy.
            sparse_label:           Whether label is an integer array instead of probability distribution.
            rc_lambda:              Tradeoff parameter for loss functions.
            from_logits:            Whether input is a log probability (usually from log_softmax) instead of unnormalized numbers.
            log_softmax_flag:       Use `F.log_softmax` when `from_logits` is `False`.
                                    If this value is `False`, this class will use not `F.log_softmax` but `F.softmax`.

            weight:                 Global scalar weight for total loss.
            classification_weight:  Global scalar weight for classification loss. If `None`, this value will be equivalent to `weight`.
            reconstruction_weight:  Global scalar weight for reconstruction loss. If `None`, this value will be equivalent to `weight`.
            grad_clip_threshold:    Threshold of the gradient clipping.
            batch_axis:             The axisthat represents mini-batch.
            **kwargs:               Any parmaeter.
        '''
        super(DRCNLoss, self).__init__(
            weight,
            batch_axis, 
            **kwargs
        )
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.__log_softmax_flag = log_softmax_flag
        self.__rc_lambda = rc_lambda
        self.__classification_weight = classification_weight
        self.__reconstruction_weight = reconstruction_weight
        self.__grad_clip_threshold = grad_clip_threshold

    def compute(
        self, 
        decoded_arr, 
        pred_arr, 
        observed_arr, 
        label_arr, 
    ):
        '''
        Compute loss.

        Args:
            decoded_arr:            `mxnet.ndarray` or `mxnet.symbol` of decoded feature points.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points.
            observed_arr:           `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data.

        Returns:
            Tensor of losses.
        '''
        return self(
            decoded_arr, 
            pred_arr, 
            observed_arr, 
            label_arr, 
        )

    def hybrid_forward(
        self, 
        F, 
        decoded_arr, 
        pred_arr, 
        observed_arr, 
        label_arr, 
        sample_weight=None
    ):
        '''
        Forward propagation, computing losses.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            decoded_arr:            `mxnet.ndarray` or `mxnet.symbol` of decoded feature points.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points.
            observed_arr:           `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data.
            sample_weight:          element-wise weighting tensor. 
                                    Must be broadcastable to the same shape as label. 
                                    For example, if label has shape (64, 10) and you want to weigh 
                                    each sample in the batch separately, sample_weight should have shape (64, 1).

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        if not self._from_logits:
            if self.__log_softmax_flag is True:
                pred_arr = F.log_softmax(pred_arr, self._axis)
            else:
                pred_arr = pred_arr - F.reshape(F.max(pred_arr, axis=self._axis), shape=(-1, 1))
                pred_arr = F.exp(pred_arr)
                pred_arr = pred_arr / F.reshape(F.sum(pred_arr, axis=self._axis), shape=(-1, 1))

        if self._sparse_label:
            classification_loss_arr = -F.pick(pred_arr, label_arr, axis=self._axis, keepdims=True)
        else:
            label_arr = _reshape_like(F, label_arr, pred_arr)
            classification_loss_arr = -F.sum(pred_arr * label_arr, axis=self._axis, keepdims=True)

        if self.__grad_clip_threshold > 0:
            classification_loss_norm = F.norm(classification_loss_arr)
            if classification_loss_norm.asscalar() > self.__grad_clip_threshold:
                classification_loss_arr = classification_loss_arr * self.__grad_clip_threshold / classification_loss_norm

        if self.__classification_weight is None:
            classification_loss_arr = _apply_weighting(F, classification_loss_arr, self._weight, sample_weight)
        else:
            classification_loss_arr = _apply_weighting(F, classification_loss_arr, self.__classification_weight, sample_weight)

        classification_loss_arr = _apply_weighting(F, classification_loss_arr, self.__rc_lambda, sample_weight)
        classification_loss = F.mean(classification_loss_arr, axis=self._batch_axis, exclude=True)

        observed_arr = _reshape_like(F, observed_arr, decoded_arr)
        reconstruction_loss_arr = F.square(observed_arr - decoded_arr)

        if self.__grad_clip_threshold > 0:
            reconstruction_loss_norm = F.norm(reconstruction_loss_arr)
            if reconstruction_loss_norm.asscalar() > self.__grad_clip_threshold:
                reconstruction_loss_arr = reconstruction_loss_arr * self.__grad_clip_threshold / reconstruction_loss_norm

        if self.__reconstruction_weight is None:
            reconstruction_loss_arr = _apply_weighting(F, reconstruction_loss_arr, self._weight / 2, sample_weight)
        else:
            reconstruction_loss_arr = _apply_weighting(F, reconstruction_loss_arr, self.__reconstruction_weight / 2, sample_weight)

        reconstruction_loss_arr = _apply_weighting(F, reconstruction_loss_arr, (1 - self.__rc_lambda), sample_weight)
        reconstruction_loss = F.mean(reconstruction_loss_arr, axis=self._batch_axis, exclude=True)

        return classification_loss + reconstruction_loss, classification_loss, reconstruction_loss
