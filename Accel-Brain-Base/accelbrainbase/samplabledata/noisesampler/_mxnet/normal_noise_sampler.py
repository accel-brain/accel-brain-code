# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler
import mxnet.ndarray as nd
import mxnet as mx


class NormalNoiseSampler(NoiseSampler):
    '''
    The class to draw fake samples from Gaussian distributions,
    generating from a `mxnet.ndarray.random`.

    References:
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).

    '''

    def __init__(
        self, 
        loc=0.0,
        scale=1.0,
        batch_size=40,
        seq_len=0,
        channel=3,
        height=96,
        width=96,
        shape_tuple=None,
        ctx=mx.cpu()
    ):
        '''
        Init.

        Args:
            loc:            `float` of mean(centre) of the distribution.
            scale:          `float` of standard deviation(spread or width) of the distribution.

            batch_size:     `int` of batch size.
            seq_len:        `int` of the length of series.
                            If this value is `0`, the rank of matrix generated is `4`. 
                            The shape is: (`batch_size`, `channel`, `height`, `width`).
                            If this value is more than `0`, the rank of matrix generated is `5`.
                            The shape is: (`batch_size`, `seq_len`, `channel`, `height`, `width`).

            channel:        `int` of channel.
            height:         `int` of image height.
            width:          `int` of image width.
            shape_tuple:    `tuple` of shape of samples.
                            If this value is not `None`,  `batch_size`, `seq_len`, `channel`, `width` and `height` will be ignored.

            ctx:            `mx.gpu` or `mx.cpu`.
        '''
        self.__loc = loc
        self.__scale = scale
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__channel = channel
        self.__height = height
        self.__width = width
        self.__shape_tuple = shape_tuple
        self.__ctx = ctx

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        if self.__shape_tuple is None:
            if self.__seq_len > 0:
                shape_tuple = (
                    self.__batch_size,
                    self.__seq_len, 
                    self.__channel,
                    self.__height,
                    self.__width
                )
            else:
                shape_tuple = (
                    self.__batch_size,
                    self.__channel, 
                    self.__height,
                    self.__width
                )
        else:
            shape_tuple = self.__shape_tuple

        observed_arr = nd.random.normal(
            loc=self.__loc, 
            scale=self.__scale, 
            shape=shape_tuple,
            ctx=self.__ctx
        )
        if self.noise_sampler is not None:
            observed_arr = observed_arr + self.noise_sampler.draw()

        return observed_arr
