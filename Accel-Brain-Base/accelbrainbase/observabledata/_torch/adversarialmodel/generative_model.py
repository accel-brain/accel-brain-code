# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.adversarial_model import AdversarialModel
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler

import torch
from torch import nn
from torch.optim.adam import Adam

import numpy as np
from logging import getLogger


class GenerativeModel(AdversarialModel):
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
        optimizer_f=None,
        condition_sampler=None,
        conditonal_dim=1,
        learning_rate=1e-05,
        ctx="cpu",
        not_init_flag=False,
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
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        super(GenerativeModel, self).__init__()

        self.noise_sampler = noise_sampler
        self.model = model
        self.__learning_rate = learning_rate
        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.condition_sampler = condition_sampler
        self.conditonal_dim = conditonal_dim
        self.__not_init_flag = not_init_flag

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if optimizer_f is not None:
                    self.optimizer = optimizer_f(
                        self.model.parameters(), 
                    )
                else:
                    self.optimizer = Adam(
                        self.model.parameters(),
                        lr=self.__learning_rate
                    )

    def draw(self):
        '''
        Draw samples from the fake distribution.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        if self.condition_sampler is None:
            return self.model(self.noise_sampler.draw())
        else:
            condition_arr, sampled_arr = self.condition_sampler.draw()
            if sampled_arr is not None:
                if self.noise_sampler is not None:
                    sampled_arr = sampled_arr + self.noise_sampler.draw()
                inferenced_arr = self.model(sampled_arr)

                generated_arr = torch.cat(
                    (
                        inferenced_arr,
                        condition_arr,
                    ),
                    dim=self.conditonal_dim
                )
            else:
                if self.noise_sampler is not None:
                    condition_arr = condition_arr + self.noise_sampler.draw()
                generated_arr = self.model(condition_arr)

            return generated_arr

    # is-a `NoiseSampler`.
    __noise_sampler = None

    def get_noise_sampler(self):
        ''' getter '''
        return self.__noise_sampler

    def set_noise_sampler(self, value):
        ''' setter '''
        if isinstance(value, NoiseSampler) is False and value is not None:
            raise TypeError("The type of `noise_sampler` must be `NoiseSampler`.")
        self.__noise_sampler = value
    
    noise_sampler = property(get_noise_sampler, set_noise_sampler)

    # is-a `ConditionSampler` of sampler to draw conditons from user-defined distributions.
    __conditon_sampler = None

    def get_condition_sampler(self):
        ''' getter for `ConditionSampler` of sampler to draw conditons from user-defined distributions.'''
        return self.__conditon_sampler
    
    def set_condition_sampler(self, value):
        ''' setter for `ConditionSampler` of sampler to draw conditons from user-defined distributions.'''
        if isinstance(value, ConditionSampler) is False:
            raise TypeError("The type of `condition_sampler` must be `ConditionSampler`.")
        self.__conditon_sampler = value

    conditon_sampler = property(get_condition_sampler, set_condition_sampler)
