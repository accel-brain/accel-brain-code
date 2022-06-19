# -*- coding: utf-8 -*-
from pycomposer.samplabledata.truesampler._torch.bar_gram_true_sampler import BarGramTrueSampler
import torch


class ConditionalBarGramTrueSampler(BarGramTrueSampler):
    '''
    Conditonal sampler which draws samples from the `true` distribution of MIDI files.
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
            `np.ndarray` of samples.
        '''
        sampled_arr = torch.cat(
            (
                super().draw(), 
                super().draw(),
            ),
            dim=self.conditonal_dim
        )
        return sampled_arr.float()
