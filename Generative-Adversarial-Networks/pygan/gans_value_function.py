# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class GANsValueFunction(metaclass=ABCMeta):
    '''
    The interface to compute rewards.
    '''

    @abstractmethod
    def compute_discriminator_reward(
        self,
        true_posterior_arr,
        generated_posterior_arr
    ):
        '''
        Compute discriminator's reward.

        Args:
            true_posterior_arr:         `np.ndarray` of `true` posterior inferenced by the discriminator.
            generated_posterior_arr:    `np.ndarray` of `fake` posterior inferenced by the discriminator.
        
        Returns:
            `np.ndarray` of Gradients.
        '''
        raise NotImplementedError()

    @abstractmethod
    def compute_generator_reward(
        self,
        generated_posterior_arr
    ):
        '''
        Compute generator's reward.

        Args:
            generated_posterior_arr:    `np.ndarray` of `fake` posterior inferenced by the discriminator.
        
        Returns:
            `np.ndarray` of Gradients.
        '''
        raise NotImplementedError()
