# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.true_sampler import TrueSampler


class ConditionalTrueSampler(TrueSampler):
    '''
    The class to draw true samples from distributions.

    You should use this class when you want to build the Condtional GAN.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        return super().draw(), super().draw()
