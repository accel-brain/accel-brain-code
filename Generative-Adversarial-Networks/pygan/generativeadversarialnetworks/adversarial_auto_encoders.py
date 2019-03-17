# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
from pygan.generative_adversarial_networks import GenerativeAdversarialNetworks
from pygan.true_sampler import TrueSampler
from pygan.generativemodel.auto_encoder_model import AutoEncoderModel
from pygan.discriminative_model import DiscriminativeModel
from pygan.gans_value_function import GANsValueFunction
from pygan.gansvaluefunction.mini_max import MiniMax


class AdversarialAutoEncoders(GenerativeAdversarialNetworks):
    '''
    The controller for the Adversarial Auto-Encoders(AAEs).
    '''

    def __init__(self, gans_value_function=None):
        '''
        Init.

        Args:
            gans_value_function:        is-a `GANsValueFunction`.
        '''
        if gans_value_function is None:
            gans_value_function = MiniMax()

        if isinstance(gans_value_function, GANsValueFunction) is False:
            raise TypeError("The type of `gans_value_function` must be `GANsValueFunction`.")
        self.__gans_value_function = gans_value_function
        self.__logger = getLogger("pygan")

        super().__init__(gans_value_function)

    def train(
        self,
        true_sampler,
        generative_model,
        discriminative_model,
        iter_n=100,
        k_step=10
    ):
        '''
        Train.

        Args:
            true_sampler:           Sampler which draws samples from the `true` distribution.
            generative_model:       Generator which draws samples from the `fake` distribution.
            discriminative_model:   Discriminator which discriminates `true` from `fake`.
            iter_n:                 The number of training iterations.
            k_step:                 The number of learning of the discriminative_model.
        
        Returns:
            Tuple data.
            - trained Generator which is-a `GenerativeModel`.
            - trained Discriminator which is-a `DiscriminativeModel`.
        '''
        if isinstance(true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")
        if isinstance(generative_model, AutoEncoderModel) is False:
            raise TypeError("The type of `generative_model` must be `AutoEncoderModel`.")
        if isinstance(discriminative_model, DiscriminativeModel) is False:
            raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        a_logs_list = []
        d_logs_list = []
        g_logs_list = []
        try:
            for n in range(iter_n):
                self.__logger.debug("-" * 100)
                self.__logger.debug("Iterations: (" + str(n+1) + "/" + str(iter_n) + ")")
                self.__logger.debug("-" * 100)
                self.__logger.debug(
                    "The `auto_encoder`'s turn."
                )
                self.__logger.debug("-" * 100)

                generative_model, a_logs_list = self.train_auto_encoder(
                    generative_model,
                    a_logs_list
                )

                self.__logger.debug("-" * 100)
                self.__logger.debug(
                    "The `discriminator`'s turn."
                )
                self.__logger.debug("-" * 100)

                discriminative_model, d_logs_list = self.train_discriminator(
                    k_step,
                    true_sampler,
                    generative_model,
                    discriminative_model,
                    d_logs_list
                )

                self.__logger.debug("-" * 100)
                self.__logger.debug(
                    "The `generator`'s turn."
                )
                self.__logger.debug("-" * 100)

                generative_model, g_logs_list = self.train_generator(
                    generative_model,
                    discriminative_model,
                    g_logs_list
                )

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

        self.__logs_tuple = (a_logs_list, d_logs_list, g_logs_list)
        return generative_model, discriminative_model

    def train_auto_encoder(self, generative_model, a_logs_list):
        '''
        Train the generative model as the Auto-Encoder.

        Args:
            generative_model:   Generator which draws samples from the `fake` distribution.
            a_logs_list:        `list` of the reconstruction errors.
        
        Returns:
            The tuple data. The shape is...
            - Generator which draws samples from the `fake` distribution.
            - `list` of the reconstruction errors.
        '''
        error_arr = generative_model.update()
        if error_arr.ndim > 1:
            error_arr = error_arr.mean()
        a_logs_list.append(error_arr)

        self.__logger.debug("The reconstruction error (mean): " + str(error_arr))

        return generative_model, a_logs_list

    def extract_logs_tuple(self):
        '''
        Extract update logs data.

        Returns:
            The shape is:
            - `list` of the reconstruction errors.
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `discriminator`'s update turn.
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `generator`'s update turn.
        '''
        return self.__logs_tuple
