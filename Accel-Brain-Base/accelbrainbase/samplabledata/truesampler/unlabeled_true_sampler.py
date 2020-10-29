# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplable_data import SamplableData
from accelbrainbase.iteratabledata.unlabeled_image_iterator import UnlabeledImageIterator


class UnlabeledTrueSampler(TrueSampler):
    '''
    The class to draw true labeled samples from distributions.

    References:
        - Bousmalis, K., Silberman, N., Dohan, D., Erhan, D., & Krishnan, D. (2017). Unsupervised pixel-level domain adaptation with generative adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3722-3731).
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.

    '''
    # is-a `ImageIterator`.
    __unlabeled_image_iterator = None

    def get_unlabeled_image_iterator(self):
        ''' getter '''
        if isinstance(self.__unlabeled_image_iterator, UnlabeledImageIterator) is False:
            raise TypeError("The type of `__unlabeled_image_iterator` must be `UnlabeledImageIterator`.")
        return self.__unlabeled_image_iterator

    def set_unlabeled_image_iterator(self, value):
        ''' setter '''
        if isinstance(value, UnlabeledImageIterator) is False:
            raise TypeError("The type of `__unlabeled_image_iterator` must be `UnlabeledImageIterator`.")
        self.__unlabeled_image_iterator = value
    
    unlabeled_image_iterator = property(get_unlabeled_image_iterator, set_unlabeled_image_iterator)

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr = None
        for arr_tuple in self.unlabeled_image_iterator.generate_learned_samples():
            observed_arr = arr_tuple[0]
            break
        return observed_arr
