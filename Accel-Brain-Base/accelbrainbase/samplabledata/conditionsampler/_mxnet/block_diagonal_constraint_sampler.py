# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler
import mxnet.ndarray as nd
import mxnet as mx


class BlockDiagonalConstraintSampler(ConditionSampler):
    '''
    The class to draw conditional samples from distributions of
    the block diagonal constraint.

    References:
        - Ghasedi, K., Wang, X., Deng, C., & Huang, H. (2019). Balanced self-paced learning for generative adversarial clustering network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4391-4400).
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    def __init__(
        self, 
        cluster_n,
        low=0.0,
        high=1.0,
        batch_size=40,
        ctx=mx.cpu()
    ):
        '''
        Init.

        Args:
            cluster_n:      `int` of the number of clusters.
            low:            `float` of lower boundary of the output interval.
                            All values generated will be greater than or equal to `low`.

            high:           `float` of upper boundary of the output interval.
                            All values generated will be less than or equal to `high`.

            batch_size:     `int` of batch size.
            ctx:            `mx.gpu` or `mx.cpu`.
        '''
        self.__cluster_n = cluster_n
        self.__low = low
        self.__high = high
        self.__batch_size = batch_size
        self.__ctx = ctx

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        observed_arr = nd.zeros((self.__batch_size, 1, self.__cluster_n, self.__cluster_n), ctx=self.__ctx)
        observed_arr[:, 0] = nd.eye(self.__cluster_n, ctx=self.__ctx)

        observed_arr = observed_arr + nd.random.uniform(
            low=self.__low, 
            high=self.__high, 
            shape=(self.__batch_size, 1, self.__cluster_n, self.__cluster_n),
            ctx=self.__ctx
        )

        if self.noise_sampler is not None:
            observed_arr = observed_arr + self.noise_sampler.draw()

        return observed_arr

    # is-a `NoiseSampler`.
    __noise_sampler = None

    def get_noise_sampler(self):
        ''' getter '''
        return self.__noise_sampler
    
    def set_noise_sampler(self, value):
        ''' setter '''
        if isinstance(value, NoiseSampler) is False:
            raise TypeError("The type of `noise_sampler` must be `NoiseSampler`.")
        self.__noise_sampler = value

    noise_sampler = property(get_noise_sampler, set_noise_sampler)
