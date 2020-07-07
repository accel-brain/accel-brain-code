# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ComputableLoss(metaclass=ABCMeta):
    '''
    The interface of Loss function.
    '''

    @abstractmethod
    def compute(
        self,
        pred_arr, 
        real_arr,
    ):
        '''
        Compute loss.

        Args:
            pred_arr:       Inferenced results.
            real_arr:       Real results.
        
        Returns:
            Tensor of losses.
        '''
        raise NotImplementedError()
