# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pygan.true_sampler import TrueSampler


class ConditionalTrueSampler(TrueSampler):
    '''
    Sampler which draws samples from the conditional `true` distribution.
    '''

    @abstractmethod
    def add_condition(self, observed_arr):
        '''
        Add condtion.

        Args:
            observed_arr:       `np.ndarray` of samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        raise NotImplementedError()
