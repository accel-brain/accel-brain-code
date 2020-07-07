# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.truesampler.labeled_true_sampler import LabeledTrueSampler


class ConditionalLabeledTrueSampler(LabeledTrueSampler):
    '''
    The class to draw true conditional labeled samples from distributions.

    References:
        - Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., & Krishnan, D. (2017). Unsupervised pixel-level domain adaptation with generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3722-3731).
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr, labeled_arr = super().draw()
        condition_arr, _ = super().draw()
        return observed_arr, condition_arr, labeled_arr
