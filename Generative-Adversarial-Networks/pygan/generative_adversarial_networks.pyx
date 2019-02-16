# -*- coding: utf-8 -*-
from pygan.true_sampler import TrueSampler
from pygan.generative_model import GenerativeModel
from pygan.discriminative_model import DiscriminativeModel


class GenerativeAdversarialNetworks(object):
    '''
    The controller for the Generative Adversarial Networks(GANs).
    '''

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
        if isinstance(generative_model, GenerativeModel) is False:
            raise TypeError("The type of `generative_model` must be `GenerativeModel`.")
        if isinstance(discriminative_model, DiscriminativeModel) is False:
            raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        for n in range(iter_n):
            for k in range(k_step):
                true_arr = true_sampler.draw()
                generated_arr = generative_model.draw()
                true_d_arr = discriminative_model.inference(true_arr)
                generated_d_arr = discriminative_model.inference(generated_arr)
                grad_arr = np.log(true_d_arr) + np.log(1 - generated_d_arr)
                discriminative_model.learn(grad_arr)

            generated_arr = generative_model.draw()
            generated_d_arr = discriminative_model.inference(generated_arr)
            grad_arr = np.log(1 - generated_d_arr)
            generative_model.learn(grad_arr)

        return generative_model, discriminative_model
