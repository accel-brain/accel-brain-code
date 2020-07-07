# -*- coding: utf-8 -*-
from accelbrainbase.samplable_data import SamplableData
from accelbrainbase.samplabledata.truesampler.labeled_true_sampler import LabeledTrueSampler


class LabeledConditionSampler(SamplableData):
    '''
    The class to draw conditional samples from distributions,
    using a `TrueSampler`.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''

    __true_sampler = None

    def get_true_sampler(self):
        ''' getter '''
        if isinstance(self.__true_sampler, LabeledTrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `LabeledTrueSampler`.")
        return self.__true_sampler
    
    def set_true_sampler(self, value):
        ''' setter '''
        if isinstance(value, LabeledTrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `LabeledTrueSampler`.")
        self.__true_sampler = value

    true_sampler = property(get_true_sampler, set_true_sampler)

    __observable_data = None

    def get_observable_data(self):
        ''' getter '''
        return self.__observable_data

    def set_observable_data(self, value):
        ''' setter '''
        self.__observable_data = value
    
    observable_data = property(get_observable_data, set_observable_data)

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        condition_arr, label_arr = self.true_sampler.draw()
        if self.observable_data is not None:
            sampled_arr = self.observable_data.inference(condition_arr)
        else:
            sampled_arr = None
        return condition_arr, sampled_arr, label_arr
