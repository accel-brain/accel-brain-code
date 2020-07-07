# -*- coding: utf-8 -*-
from accelbrainbase.samplable_data import SamplableData
from accelbrainbase.iteratable_data import IteratableData


class NoiseSampler(SamplableData):
    '''
    The abstract class to draw fake samples from distributions,
    generating from `IteratableData`.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
        - Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., & Krishnan, D. (2017). Unsupervised pixel-level domain adaptation with generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3722-3731).

    '''

    # is-a `NoiseSampler`.
    __noise_sampler = None

    def get_noise_sampler(self):
        ''' getter for `NoiseSampler`.'''
        return self.__noise_sampler
    
    def set_noise_sampler(self, value):
        ''' setter for `NoiseSampler`. '''
        self.__noise_sampler = value

    noise_sampler = property(get_noise_sampler, set_noise_sampler)

    # is-a `IteratorableData`.
    __iteratorable_data = None

    def get_iteratorable_data(self):
        ''' getter for `IteratableData`.'''
        if isinstance(self.__iteratorable_data, IteratorableData) is False:
            raise TypeError("The type of `__iteratorable_data` must be `IteratorableData`.")
        return self.__iteratorable_data
    
    def set_iteratorable_data(self, value):
        ''' setter for `IteratableData`.'''
        if isinstance(value, IteratorableData) is False:
            raise TypeError("The type of `__iteratorable_data` must be `IteratorableData`.")
        self.__iteratorable_data = value
    
    iteratorable_data = property(get_iteratorable_data, set_iteratorable_data)

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr = None
        for arr_tuple in self.iteratorable_data.generate_learned_samples():
            observed_arr = arr_tuple[0]
            break

        if self.noise_sampler is not None:
            observed_arr = observed_arr + self.noise_sampler.draw()

        return observed_arr
