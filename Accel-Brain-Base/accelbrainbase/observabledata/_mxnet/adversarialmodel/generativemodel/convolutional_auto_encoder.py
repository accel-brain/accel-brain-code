# -*- coding: utf-8 -*-
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder as _ConvolutionalAutoEncoder

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger


class ConvolutionalAutoEncoder(GenerativeModel):
    '''
    Generative model, which draws samples from the fake distribution, 
    in the Generative Adversarial Networks(GANs).

    The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes 
    a min-max adversarial game between two neural networks – a generative model, G, and a 
    discriminative model, D. The discriminator model, D(x), is a neural network that computes 
    the probability that a observed data point x in data space is a sample from the data 
    distribution (positive samples) that we are trying to model, rather than a sample from our 
    generative model (negative samples). 
    
    Concurrently, the generator uses a function G(z) that maps samples z from the prior p(z) to 
    the data space. G(z) is trained to maximally confuse the discriminator into believing that 
    samples it generates come from the data distribution. The generator is trained by leveraging 
    the gradient of D(x) w.r.t. x, and using that to modify its parameters.

    References:
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
        - Warde-Farley, D., & Bengio, Y. (2016). Improving generative adversarial networks with denoising feature matching.

    '''

    def __init__(
        self, 
        noise_sampler, 
        model, 
        initializer=None,
        condition_sampler=None,
        conditonal_dim=1,
        learning_rate=1e-05,
        optimizer_name="SGD",
        hybridize_flag=True,
        scale=1.0, 
        ctx=mx.gpu(), 
        **kwargs
    ):
        '''
        Init.

        Args:
            noise_sampler:                  is-a `NoiseSampler`.
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            condition_sampler:              is-a `ConditionSampler` of sampler to draw conditons from user-defined distributions.
            conditonal_dim:                 `int` of the dimension to be concated conditions and observed data points.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        if isinstance(model, _ConvolutionalAutoEncoder) is False:
            raise TypeError("The type of `model` must be `accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder.ConvolutionalAutoEncoder`.")

        super().__init__(
            noise_sampler=noise_sampler, 
            model=model, 
            initializer=initializer,
            condition_sampler=condition_sampler,
            conditonal_dim=conditonal_dim,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            hybridize_flag=hybridize_flag,
            scale=scale, 
            ctx=ctx, 
            **kwargs
        )

        self.condition_sampler = condition_sampler
        self.conditonal_dim = conditonal_dim

    def draw(self):
        '''
        Draw samples from the fake distribution.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        if self.condition_sampler is None:
            observed_arr = self.noise_sampler.draw()
            encoded_arr = self.model.encoder(observed_arr)
            decoded_arr = self.model.decoder(encoded_arr)
            generated_arr = encoded_arr
        else:
            condition_arr, sampled_arr = self.condition_sampler.draw()
            if sampled_arr is not None:
                sampled_arr = sampled_arr + self.noise_sampler.draw()
                encoded_arr = self.model.encoder(sampled_arr)
                decoded_arr = self.model.decoder(encoded_arr)

                observed_arr = sampled_arr

                generated_arr = nd.concat(
                    encoded_arr,
                    condition_arr,
                    dim=self.conditonal_dim
                )
            else:
                condition_arr = condition_arr + self.noise_sampler.draw()
                encoded_arr = self.model.encoder(condition_arr)
                decoded_arr = self.model.decoder(encoded_arr)

                observed_arr = condition_arr
                generated_arr = encoded_arr

        return generated_arr, observed_arr, decoded_arr

    # is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
    __model = None

    def get_model(self):
        ''' getter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        return self.__model
    
    def set_model(self, value):
        ''' setter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`.'''
        if isinstance(value, _ConvolutionalAutoEncoder) is False:
            raise TypeError("The type of `model` must be `accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder.ConvolutionalAutoEncoder`.")

        self.__model = value
    
    model = property(get_model, set_model)
