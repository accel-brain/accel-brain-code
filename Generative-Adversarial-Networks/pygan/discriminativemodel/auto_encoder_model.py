# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pygan.discriminative_model import DiscriminativeModel


class AutoEncoderModel(DiscriminativeModel):
    '''
    Auto-Encoder as a Discriminative Model
    which discriminates `true` from `fake`.
    '''

    @abstractmethod
    def pre_learn(self, true_sampler, epochs=1000):
        '''
        Pre learning.

        Args:
            true_sampler:       is-a `TrueSampler`.
            epochs:             Epochs.
        '''
        raise NotImplementedError()
