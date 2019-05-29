# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler
from pygan.discriminative_model import DiscriminativeModel
from pydbm.loss.mean_squared_error import MeanSquaredError


class FeatureMatching(object):
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

    def __init__(self, lambda1=1.0, lambda2=0.0):
        '''
        Init.

        Args:
            lambda1:        Weight for results of standard feature matching.
            lambda2:        Weight for results of difference between generated data points and true samples.

        Exceptions:
            ValueError:     When the sum of `lambda1` and `lambda2` is not less than `1.0`.
                            Those parameters are trade-off parameters.
        '''
        self.__true_arr = None
        self.__mean_squared_error = MeanSquaredError()
        self.__loss_list = []

        if lambda1 + lambda2 > 1:
            raise ValueError("The sum of `lambda1` and `lambda2` must be less than `1.0`. Those parameters are trade-off parameters.")
        self.__lambda1 = lambda1
        self.__lambda2 = lambda2

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
        if isinstance(true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")

        if isinstance(discriminative_model, DiscriminativeModel) is False:
            raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        self.__true_arr = true_sampler.draw()

        grad_arr1 = np.zeros_like(generated_arr)
        grad_arr2 = np.zeros_like(generated_arr)
        loss1, loss2 = 0.0, 0.0
        if self.__lambda1 > 0.0:
            _generated_arr = discriminative_model.feature_matching_forward(generated_arr)
            _true_arr = discriminative_model.feature_matching_forward(self.__true_arr)

            grad_arr1 = self.__mean_squared_error.compute_delta(
                _generated_arr,
                _true_arr
            )
            grad_arr1 = discriminative_model.feature_matching_backward(grad_arr1)
            grad_arr1 = grad_arr1.reshape(generated_arr.shape)

            loss1 = self.__mean_squared_error.compute_loss(
                _generated_arr,
                _true_arr
            )
        if self.__lambda2 > 0.0:
            grad_arr2 = self.__mean_squared_error.compute_delta(
                generated_arr,
                self.__true_arr
            )
            loss2 = self.__mean_squared_error.compute_loss(
                generated_arr,
                self.__true_arr
            )
        grad_arr = (grad_arr1 * self.__lambda1) + (grad_arr2 * self.__lambda2)
        loss = (loss1 * self.__lambda1) + (loss2 * self.__lambda2)

        self.__loss_list.append(loss)
        return grad_arr

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_true_arr(self):
        ''' getter '''
        return self.__true_arr

    true_arr = property(get_true_arr, set_readonly)

    def get_mean_squared_error(self):
        ''' getter '''
        return self.__mean_squared_error
    
    mean_squared_error = property(get_mean_squared_error, set_readonly)

    def get_loss_arr(self):
        ''' getter '''
        return np.array(self.__loss_list)
    
    loss_arr = property(get_loss_arr, set_readonly)
