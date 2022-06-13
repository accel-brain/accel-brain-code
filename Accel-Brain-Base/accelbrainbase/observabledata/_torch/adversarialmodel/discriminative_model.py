# -*- coding: utf-8 -*-
from accelbrainbase.observabledata._torch.adversarial_model import AdversarialModel
import numpy as np
import torch
from torch import nn
from logging import getLogger
from torch.optim.adam import Adam


class DiscriminativeModel(AdversarialModel):
    '''
    Discriminative model, which discriminates true from fake,
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
        model, 
        optimizer_f=None,
        learning_rate=1e-05,
        ctx="cpu", 
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        super(DiscriminativeModel, self).__init__()

        self.model = model
        self.__learning_rate = learning_rate
        logger = getLogger("accelbrainbase")
        self.__logger = logger

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

    def inference(self, observed_arr):
        '''
        Draw samples from the fake distribution.

        Args:
            observed_arr:       `mxnet.ndarray` or `mxnet.symbol` of observed data points.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        return self.forward(observed_arr)

    def forward(self, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        return self.model(x)
