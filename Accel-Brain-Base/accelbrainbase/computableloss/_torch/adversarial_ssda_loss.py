# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss

import torch
import torch.nn as nn

import numpy as np


class AdversarialSSDALoss(nn.modules.loss._Loss, ComputableLoss):
    '''
    Loss function of Self-supervised domain adaptation.

    References:
        - Jing, L., & Tian, Y. (2020). Self-supervised visual feature learning with deep neural networks: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence.
        - Xu, J., Xiao, L., & López, A. M. (2019). Self-supervised domain adaptation for computer vision tasks. IEEE Access, 7, 156694-156706., p156698.
    '''

    def __init__(
        self, 
        weight=1.0,
        classification_loss_f=None,
        pretext_loss_f=None,
        classification_weight=None, 
        pretext_weight=None, 
        adversarial_weight=None,
        adversarial_lambda=0.5,
        batch_axis=1, 
        label_smoothing=0.0,
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
            adversarial_weight:     Global scalar weight for adversarial loss. If `None`, this value will be equivalent to `weight`.
            adversarial_lambda:     Trade-off parameter for adversarial loss. $L_{d} = (1 - `adversarial_lambda`) * {target_posterior_arr} + `adversarial_lambda` * {source_posterior_arr}.
            grad_clip_threshold:    Threshold of the gradient clipping.
            batch_axis:             The axisthat represents mini-batch.
            **kwargs:               Any parmaeter.
        '''
        super(AdversarialSSDALoss, self).__init__()

        if classification_loss_f is not None:
            self.__classification_loss_f = classification_loss_f
        else:
            self.__classification_loss_f = torch.nn.CrossEntropyLoss(
                size_average=None, 
                ignore_index=-100, 
                reduce=None, 
                reduction='mean', 
                label_smoothing=label_smoothing
            )

        if pretext_loss_f is not None:
            self.__pretext_loss_f = pretext_loss_f
        else:
            self.__pretext_loss_f = torch.nn.CrossEntropyLoss(
                size_average=None, 
                ignore_index=-100, 
                reduce=None, 
                reduction='mean', 
                label_smoothing=label_smoothing
            )

        self.__weight = weight
        self.__classification_weight = classification_weight
        self.__pretext_weight = pretext_weight
        self.__adversarial_weight = adversarial_weight
        self.__adversarial_lambda = adversarial_lambda
        self.__batch_axis = batch_axis

    def compute(
        self, 
        pretext_pred_arr, 
        pred_arr, 
        pretext_label_arr, 
        label_arr, 
        source_posterior_arr,
        target_posterior_arr,
    ):
        '''
        Compute loss.

        Args:
            pretext_pred_arr:       `mxnet.ndarray` or `mxnet.symbol` of decoded feature points.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points.
            pretext_label_arr:      `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data.
            source_posterior_arr:     `mxnet.ndarray` or `mxnet.symbol` of encoded data points in source domain.
            target_posterior_arr:     `mxnet.ndarray` or `mxnet.symbol` of encoded data poitns in target domain.

        Returns:
            Tensor of losses.
        '''
        return self(
            pretext_pred_arr, 
            pred_arr, 
            pretext_label_arr, 
            label_arr, 
            source_posterior_arr,
            target_posterior_arr,
        )

    def forward(
        self, 
        pretext_pred_arr, 
        pred_arr, 
        pretext_label_arr, 
        label_arr, 
        source_posterior_arr,
        target_posterior_arr,
    ):
        '''
        Forward propagation, computing losses.

        Args:
            F:                      `mxnet.ndarray` or `mxnet.symbol`.
            pretext_pred_arr:       `mxnet.ndarray` or `mxnet.symbol` of decoded feature points.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points.
            pretext_label_arr:      `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data.
            source_posterior_arr:     `mxnet.ndarray` or `mxnet.symbol` of encoded data points in source domain.
            target_posterior_arr:     `mxnet.ndarray` or `mxnet.symbol` of encoded data poitns in target domain.

            sample_weight:          element-wise weighting tensor. 
                                    Must be broadcastable to the same shape as label. 
                                    For example, if label has shape (64, 10) and you want to weigh 
                                    each sample in the batch separately, sample_weight should have shape (64, 1).

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        classification_loss = self.__classification_loss_f(
            pred_arr,
            label_arr
        )
        if self.__classification_weight is not None:
            classification_loss = classification_loss * self.__classification_weight

        classification_loss = classification_loss * self.__weight

        pretext_label_arr = pretext_label_arr.reshape_as(pretext_pred_arr)

        pretext_loss = self.__pretext_loss_f(
            pretext_pred_arr,
            pretext_label_arr
        )
        if self.__pretext_weight is not None:
            pretext_loss = pretext_loss * self.__pretext_weight

        pretext_loss = pretext_loss * self.__weight

        adversarial_loss_arr = - ((1 - self.__adversarial_lambda) * torch.log(target_posterior_arr) + self.__adversarial_lambda * torch.log(source_posterior_arr))
        adversarial_loss_arr = torch.reshape(adversarial_loss_arr, (adversarial_loss_arr.shape[0], -1))
        adversarial_loss = torch.mean(adversarial_loss_arr)

        if self.__adversarial_weight is not None:
            adversarial_loss = adversarial_loss * self.__adversarial_weight

        adversarial_loss = adversarial_loss * self.__weight

        total_loss = classification_loss + pretext_loss + adversarial_loss
        return total_loss, classification_loss, pretext_loss, adversarial_loss
