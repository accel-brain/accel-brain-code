# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class DiscriminativeModel(metaclass=ABCMeta):
    '''
    Discriminator which discriminates `true` from `fake`.
    '''

    @abstractmethod
    def inference(self, observed_arr):
        '''
        Draws samples from the `true` distribution.

        Args:
            observed_arr:     `np.ndarray` of observed data points.
        
        Returns:
            `np.ndarray` of inferenced.
            `0` is to `1` what `fake` is to `true`.
        '''
        raise NotImplementedError()

    @abstractmethod
    def learn(self, grad_arr, fix_opt_flag=False):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:       `np.ndarray` of gradients.
            fix_opt_flag:   If `False`, no optimization in this model will be done.        
        
        Returns:
            `np.ndarray` of delta or gradients.
        '''
        raise NotImplementedError()
