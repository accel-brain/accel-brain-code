# -*- coding: utf-8 -*-
import numpy as np
from pygan.feature_matching import FeatureMatching


class MidiNetFeatureMatching(FeatureMatching):
    '''
    Value function with Feature matching, which addresses the instability of GANs 
    by specifying a new objective for the generator that prevents it from overtraining 
    on the current discriminator(Salimans, T., et al., 2016).
    
    "Instead of directly maximizing the output of the discriminator, 
    the new objective requires the generator to generate data that matches
    the statistics of the real data, where we use the discriminator only to specify 
    the statistics that we think are worth matching." (Salimans, T., et al., 2016, p2.)

    References:
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.

    '''

    # trade-off parameters.
    __lambda1 = 0.5
    __lambda2 = 0.5

    def get_lambda1(self):
        ''' getter '''
        return self.__lambda1
    
    def set_lambda1(self, value):
        ''' setter '''
        self.__lambda1 = value
    
    lambda1 = property(get_lambda1, set_lambda1)

    def get_lambda2(self):
        ''' getter '''
        return self.__lambda2
    
    def set_lambda2(self, value):
        ''' setter '''
        self.__lambda2 = value
    
    lambda2 = property(get_lambda2, set_lambda2)

    def compute_delta(
        self,
        true_sampler, 
        discriminative_model,
        generated_arr
    ):
        '''
        Compute generator's reward.

        Args:
            true_sampler:           Sampler which draws samples from the `true` distribution.
            discriminative_model:   Discriminator which discriminates `true` from `fake`.
            generated_arr:          `np.ndarray` generated data points.
        
        Returns:
            `np.ndarray` of Gradients.
        '''
        grad_arr1 = super().compute_delta(
            true_sampler,
            discriminative_model,
            generated_arr
        )
        grad_arr2 = self.mean_squared_error.compute_delta(generated_arr, self.true_arr)
        grad_arr = (grad_arr1 * self.__lambda1) + (grad_arr2 * self.__lambda2)
        return grad_arr
