# -*- coding: utf-8 -*-
from pycomposer.samplabledata.noisesampler.bar_gram_noise_sampler import BarGramNoiseSampler as _BarGramNoiseSampler
import torch


class BarGramNoiseSampler(_BarGramNoiseSampler):
    '''
    Sampler which draws samples from the `true` distribution of MIDI files.
    '''

    __ctx = "cpu"

    def get_ctx(self):
        ''' getter '''
        return self.__ctx

    def set_ctx(self, value):
        ''' setter '''
        self.__ctx = value

    ctx = property(get_ctx, set_ctx)

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        arr = super().draw()
        arr = torch.from_numpy(arr)
        arr = arr.to(self.__ctx)
        return arr.float()
