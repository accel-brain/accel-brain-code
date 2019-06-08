# -*- coding: utf-8 -*-
import numpy as np
from pygan.gans_value_function import GANsValueFunction


class MarginLoss(GANsValueFunction):
    '''
    Value function in energy-based GANs framework.

    References:
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
    '''

    def __init__(
        self, 
        margin=1.0,
        margin_attenuate_rate=0.1,
        attenuate_epoch=50
    ):
        '''
        Init.

        Args:
            margin:                         margin.
            margin_attenuate_rate:          Attenuate the `margin` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `margin` by a factor of `margin_attenuate_rate` every `attenuate_epoch`.
        '''
        self.__margin = margin
        self.__margin_attenuate_rate = margin_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch
        self.__epoch = 0
        self.__discriminator_reward_list = []

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
        grad_arr = true_posterior_arr + np.maximum(0, (self.__margin - generated_posterior_arr))
        self.__epoch += 1
        if self.__epoch % self.__attenuate_epoch == 0:
            self.__margin = self.__margin * self.__margin_attenuate_rate
        self.__discriminator_reward_list.append(grad_arr.mean())
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
        grad_arr = generated_posterior_arr
        return grad_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_margin(self):
        ''' getter '''
        return self.__margin
    
    margin = property(get_margin, set_readonly)

    def get_discriminator_reward_arr(self):
        ''' getter '''
        return np.array(self.__discriminator_reward_list)

    discriminator_reward_arr = property(get_discriminator_reward_arr, set_readonly)
