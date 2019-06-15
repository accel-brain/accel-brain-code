# -*- coding: utf-8 -*-
import numpy as np
from pygan.truesampler.conditional_true_sampler import ConditionalTrueSampler
from pygan.truesampler.image_true_sampler import ImageTrueSampler


class ConditionalImageTrueSampler(ConditionalTrueSampler):
    '''
    Sampler which draws samples from the conditional `true` distribution of images.
    '''

    # The axis along which the arrays will be joined conditions and generated data.
    __conditional_axis = 1

    def __init__(self, image_true_sampler):
        '''
        Init.

        Args:
            image_true_sampler:     is-a `ImageTrueSampler`.
        '''
        if isinstance(image_true_sampler, ImageTrueSampler) is False:
            raise TypeError()
        self.__image_true_sampler = image_true_sampler

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        observed_arr = self.__image_true_sampler.draw()
        observed_arr = self.add_condition(observed_arr)
        return observed_arr

    def add_condition(self, observed_arr):
        '''
        Add condtion.

        Args:
            observed_arr:       `np.ndarray` of samples.

        Returns:
            `np.ndarray` of samples.
        '''
        if self.__image_true_sampler.seq_len is None:
            condition_arr = self.__image_true_sampler.draw()
            return np.concatenate((observed_arr, condition_arr), axis=self.conditional_axis)
        else:
            return np.concatenate(
                (
                    observed_arr[:, 0, :, :, :],
                    observed_arr[:, 1, :, :, :]
                ),
                axis=self.conditional_axis
            )

    def get_conditional_axis(self):
        ''' getter '''
        return self.__conditional_axis
    
    def set_conditional_axis(self, value):
        ''' setter '''
        self.__conditional_axis = value
    
    conditional_axis = property(get_conditional_axis, set_conditional_axis)
