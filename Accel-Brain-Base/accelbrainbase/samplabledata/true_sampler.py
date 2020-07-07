# -*- coding: utf-8 -*-
from accelbrainbase.samplable_data import SamplableData
from accelbrainbase.iteratable_data import IteratableData


class TrueSampler(SamplableData):
    '''
    The class to draw true samples from distributions,
    generating from an `IteratableData`.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''
    # is-a `IteratableData`.
    __iteratorable_data = None

    def get_iteratorable_data(self):
        ''' getter for `IteratableData`.'''
        if isinstance(self.__iteratorable_data, IteratableData) is False:
            raise TypeError("The type of `__iteratorable_data` must be `IteratableData`.")
        return self.__iteratorable_data
    
    def set_iteratorable_data(self, value):
        ''' setter for `IteratableData`.'''
        if isinstance(value, IteratableData) is False:
            raise TypeError("The type of `__iteratorable_data` must be `IteratableData`.")
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
        return observed_arr
