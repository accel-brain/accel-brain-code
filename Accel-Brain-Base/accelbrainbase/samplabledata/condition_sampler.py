# -*- coding: utf-8 -*-
from accelbrainbase.samplable_data import SamplableData
from accelbrainbase.samplabledata.true_sampler import TrueSampler


class ConditionSampler(SamplableData):
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
        if isinstance(self.__true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")
        return self.__true_sampler
    
    def set_true_sampler(self, value):
        ''' setter '''
        if isinstance(value, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")
        self.__true_sampler = value

    true_sampler = property(get_true_sampler, set_true_sampler)

    # is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
    __model = None

    def get_model(self):
        ''' getter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`.'''
        return self.__model

    def set_model(self, value):
        ''' setter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`.'''
        self.__model = value
    
    model = property(get_model, set_model)

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        condition_arr = self.true_sampler.draw()
        if self.model is not None:
            sampled_arr = self.model(condition_arr)
        else:
            sampled_arr = None
        return condition_arr, sampled_arr
