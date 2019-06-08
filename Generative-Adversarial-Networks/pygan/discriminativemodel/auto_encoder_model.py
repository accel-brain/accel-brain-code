# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
from pygan.discriminative_model import DiscriminativeModel


class AutoEncoderModel(DiscriminativeModel):
    '''
    Auto-Encoder as a Discriminative Model
    which discriminates `true` from `fake`.

    The Energy-based GAN framework considers the discriminator as an energy function, 
    which assigns low energy values to real data and high to fake data. 
    The generator is a trainable parameterized function that produces 
    samples in regions to which the discriminator assigns low energy. 

    References:
        - Manisha, P., & Gujar, S. (2018). Generative Adversarial Networks (GANs): What it can generate and What it cannot?. arXiv preprint arXiv:1804.00140.
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
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
