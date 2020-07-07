# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class SamplableData(metaclass=ABCMeta):
    '''
    The interface to draw samples from distributions.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    @abstractmethod
    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of samples.
        '''
        raise NotImplementedError()
