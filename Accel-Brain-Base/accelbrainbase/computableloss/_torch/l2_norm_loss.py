# -*- coding: utf-8 -*-
from accelbrainbase.computable_loss import ComputableLoss
import torch
import torch.nn as nn


class L2NormLoss(nn.modules.loss._Loss, ComputableLoss):
    '''
    Loss function that computes L2 norm.
    '''

    def __init__(self, weight=1.0, batch_axis=1):
        '''
        Init.

        Args:
            weight:         Global scalar weight for loss.
            batch_axis:     The axisthat represents mini-batch.
        '''
        self.__weight = weight
        self.__batch_axis = batch_axis
        super(L2NormLoss, self).__init__()

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

    def forward(self, orign_arr, dest_arr):
        '''
        Forward propagation, computing L2 norm.

        Args:
            orign_arr:   `tensor` of origins.
            dest_arr:    `tensor` of destinations.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of loss.
        '''
        dest_arr = dest_arr.reshape_as(orign_arr)
        loss = torch.sqrt(torch.mean(torch.square(orign_arr - dest_arr), dim=self.__batch_axis, keepdim=True))
        loss = loss * self.__weight
        return torch.mean(loss)
