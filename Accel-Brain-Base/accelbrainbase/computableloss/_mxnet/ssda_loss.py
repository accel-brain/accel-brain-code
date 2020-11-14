# -*- coding: utf-8 -*-
from mxnet.gluon.loss import Loss
from accelbrainbase.computable_loss import ComputableLoss

import numpy as np
from mxnet.gluon.loss import _reshape_like, _apply_weighting


class SSDALoss(Loss, ComputableLoss):
    '''
    Loss function of Self-supervised domain adaptation.

    References:
        - Jing, L., & Tian, Y. (2020). Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.
        - Xu, J., Xiao, L., & López, A. M. (2019). Self-supervised domain adaptation for computer vision tasks. IEEE Access, 7, 156694-156706., p156698.
    '''

    def __init__(
        self, 
        axis=-1,
        sparse_label=False,
        from_logits=False,
        log_softmax_flag=True,
        weight=1.0,
        classification_weight=None, 
        pretext_weight=None, 
        grad_clip_threshold=0.0,
        batch_axis=0, 
        **kwargs
    ):
        '''
        Init.

        Args:
            axis:                   The axis to sum over when computing softmax and entropy.
            sparse_label:           Whether label is an integer array instead of probability distribution.
            from_logits:            Whether input is a log probability (usually from log_softmax) instead of unnormalized numbers.
            log_softmax_flag:       Use `F.log_softmax` when `from_logits` is `False`.
                                    If this value is `False`, this class will use not `F.log_softmax` but `F.softmax`.

            weight:                 Global scalar weight for total loss.
            classification_weight:  Global scalar weight for classification loss. If `None`, this value will be equivalent to `weight`.
            pretext_weight:         Global scalar weight for pretext loss. If `None`, this value will be equivalent to `weight`.
            grad_clip_threshold:    Threshold of the gradient clipping.
            batch_axis:             The axisthat represents mini-batch.
            **kwargs:               Any parmaeter.
        '''
        super(SSDALoss, self).__init__(
            weight,
            batch_axis, 
            **kwargs
        )
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self.__log_softmax_flag = log_softmax_flag
        self.__classification_weight = classification_weight
        self.__pretext_weight = pretext_weight
        self.__grad_clip_threshold = grad_clip_threshold

    def compute(
        self, 
        pretext_pred_arr, 
        pred_arr, 
        pretext_label_arr, 
        label_arr, 
    ):
        '''
        Compute loss.

        Args:
            pretext_pred_arr:       `mxnet.ndarray` or `mxnet.symbol` of predicted data in pretext, or target domain.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points in source domain.
            pretext_label_arr:      `mxnet.ndarray` or `mxnet.symbol` of label data in pretext.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data in source domain.

        Returns:
            Tensor of losses.
        '''
        return self(
            pretext_pred_arr, 
            pred_arr, 
            pretext_label_arr, 
            label_arr, 
        )

    def hybrid_forward(
        self, 
        F, 
        pretext_pred_arr, 
        pred_arr, 
        pretext_label_arr, 
        label_arr, 
        sample_weight=None
    ):
        '''
        Forward propagation, computing losses.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            pretext_pred_arr:       `mxnet.ndarray` or `mxnet.symbol` of predicted data in pretext, or target domain.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points in source domain.
            pretext_label_arr:      `mxnet.ndarray` or `mxnet.symbol` of label data in pretext.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data in source domain.

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

        pretext_label_arr = _reshape_like(F, pretext_label_arr, pretext_pred_arr)
        pretext_loss_arr = -F.sum(pretext_pred_arr * pretext_label_arr, axis=self._axis, keepdims=True) / 4

        if self.__grad_clip_threshold > 0:
            pretext_loss_norm = F.norm(pretext_loss_arr)
            if pretext_loss_norm.asscalar() > self.__grad_clip_threshold:
                pretext_loss_arr = pretext_loss_arr * self.__grad_clip_threshold / pretext_loss_norm

        if self.__classification_weight is None:
            classification_loss_arr = _apply_weighting(F, classification_loss_arr, self._weight, sample_weight)
        else:
            classification_loss_arr = _apply_weighting(F, classification_loss_arr, self.__classification_weight, sample_weight)

        if self.__pretext_weight is None:
            pretext_loss_arr = _apply_weighting(F, pretext_loss_arr, self._weight, sample_weight)
        else:
            pretext_loss_arr = _apply_weighting(F, pretext_loss_arr, self.__pretext_weight, sample_weight)

        classification_loss = F.mean(classification_loss_arr, axis=self._batch_axis, exclude=True)
        pretext_loss = F.mean(pretext_loss_arr, axis=self._batch_axis, exclude=True)

        total_loss = classification_loss + pretext_loss
        return total_loss, classification_loss, pretext_loss
