# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from accelbrainbase.computable_loss import ComputableLoss
from accelbrainbase.computableloss._torch.l2_norm_loss import L2NormLoss


class DRCNLoss(nn.modules.loss._Loss, ComputableLoss):
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
        rc_lambda=0.75,
        weight=1.0,
        label_smoothing=0.0,
        classification_weight=None, 
        reconstruction_weight=None, 
        batch_axis=1, 
    ):
        '''
        Init.

        Args:
            axis:                   The axis to sum over when computing softmax and entropy.
            sparse_label:           Whether label is an integer array instead of probability distribution.
            rc_lambda:              Tradeoff parameter for loss functions.
            from_logits:            Whether input is a log probability (usually from log_softmax) instead of unnormalized numbers.
            weight:                 Global scalar weight for total loss.
            classification_weight:  Global scalar weight for classification loss. If `None`, this value will be equivalent to `weight`.
            reconstruction_weight:  Global scalar weight for reconstruction loss. If `None`, this value will be equivalent to `weight`.
            batch_axis:             The axisthat represents mini-batch.
        '''
        super(DRCNLoss, self).__init__()
        self.__cross_entropy_loss = torch.nn.CrossEntropyLoss(
            size_average=None, 
            ignore_index=-100, 
            reduce=None, 
            reduction='mean', 
            label_smoothing=label_smoothing
        )
        self.__l2_norm_loss = L2NormLoss(batch_axis=batch_axis)
        self.__classification_weight = classification_weight
        self.__reconstruction_weight = reconstruction_weight
        self.__weight = weight
        self.__rc_lambda = rc_lambda

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

    def forward(
        self, 
        decoded_arr, 
        pred_arr, 
        observed_arr, 
        label_arr, 
    ):
        '''
        Forward propagation, computing losses.

        Args:
            decoded_arr:            `mxnet.ndarray` or `mxnet.symbol` of decoded feature points.
            pred_arr:               `mxnet.ndarray` or `mxnet.symbol` of inferenced labeled feature points.
            observed_arr:           `mxnet.ndarray` or `mxnet.symbol` of observed data points.
            label_arr:              `mxnet.ndarray` or `mxnet.symbol` of label data.

        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        classification_loss = self.__cross_entropy_loss(
            pred_arr,
            label_arr
        )

        if self.__classification_weight is not None:
            classification_loss = classification_loss * self.__classification_weight

        observed_arr = observed_arr.reshape_as(decoded_arr)
        reconstruction_loss = self.__l2_norm_loss(observed_arr, decoded_arr)

        if self.__reconstruction_weight is not None:
            reconstruction_loss = reconstruction_loss * self.__reconstruction_weight

        classification_loss = classification_loss * self.__rc_lambda
        reconstruction_loss = reconstruction_loss * (1 - self.__rc_lambda)
        if self.__weight is not None:
            classification_loss = classification_loss * self.__weight
            reconstruction_loss = reconstruction_loss * self.__weight

        return classification_loss + reconstruction_loss, classification_loss, reconstruction_loss
