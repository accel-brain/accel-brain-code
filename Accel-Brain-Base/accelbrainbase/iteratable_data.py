# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class IteratableData(metaclass=ABCMeta):
    '''
    The interface to draw mini-batch samples from distributions.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    @abstractmethod
    def generate_learned_samples(self):
        '''
        Draw and generate learned samples.

        Returns:
            `Tuple` data. The shape is ...
            - observed data points in training.
            - supervised data in training.
            - observed data points in test.
            - supervised data in test.
        '''
        raise NotImplementedError()

    @abstractmethod
    def generate_inferenced_samples(self):
        '''
        Draw and generate inferenced samples.

        Returns:
            - observed data points in training.
        '''
        raise NotImplementedError()
