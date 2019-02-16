# -*- coding: utf-8 -*-
import numpy as np
from pygan.gans_value_function import GANsValueFunction


class MiniMax(GANsValueFunction):
    '''
    Value function in GANs framework.
    '''

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
        grad_arr = np.log(true_posterior_arr + 1e-08) + np.log(1 - generated_posterior_arr + 1e-08)
        return grad_arr

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
        grad_arr = np.log(1 - generated_posterior_arr + 1e-08)
        return grad_arr
