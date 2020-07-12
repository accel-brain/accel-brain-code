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

    __conditonal_dim = 1

    def get_conditonal_dim(self):
        ''' getter '''
        return self.__conditonal_dim

    def set_conditonal_dim(self, value):
        ''' setter '''
        self.__conditonal_dim = value

    conditonal_dim = property(get_conditonal_dim, set_conditonal_dim)

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `nd.ndarray` of samples.
        '''

        return nd.concat(
            super().draw(), 
            super().draw(),
            dim=self.conditonal_dim
        )
