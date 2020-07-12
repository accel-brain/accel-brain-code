# -*- coding: utf-8 -*-
from pycomposer.samplabledata.truesampler.bar_gram_true_sampler import BarGramTrueSampler as _BarGramTrueSampler
import mxnet as mx
import mxnet.ndarray as nd


class BarGramTrueSampler(_BarGramTrueSampler):
    '''
    Sampler which draws samples from the `true` distribution of MIDI files.
    '''

    __ctx = mx.gpu()

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
        arr = nd.ndarray.array(arr, ctx=self.__ctx)
        return arr
