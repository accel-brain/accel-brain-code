# -*- coding: utf-8 -*-
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError
from accelbrainbase.observabledata._mxnet.adversarial_model import AdversarialModel

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class DiscriminativeModel(HybridBlock, AdversarialModel):
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
        initializer=None,
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
            model:                          is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            learning_rate:                  `float` of learning rate.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        super(DiscriminativeModel, self).__init__(**kwargs)

        self.model = model
        logger = getLogger("accelbrainbase")
        self.__logger = logger

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=2
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.trainer = gluon.Trainer(
                    self.collect_params(),
                    optimizer_name,
                    {
                        "learning_rate": learning_rate
                    }
                )
                if hybridize_flag is True:
                    self.model.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.model.collect_params(select)
        return params_dict

    def inference(self, observed_arr):
        '''
        Draw samples from the fake distribution.

        Args:
            observed_arr:       `mxnet.ndarray` or `mxnet.symbol` of observed data points.

        Returns:
            `Tuple` of `mxnet.ndarray`s.
        '''
        return self(observed_arr)

    def hybrid_forward(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        # rank-3
        return self.forward_propagation(F, x)

    def forward_propagation(self, F, x):
        '''
        Hybrid forward with Gluon API.

        Args:
            F:      `mxnet.ndarray` or `mxnet.symbol`.
            x:      `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` or `mxnet.symbol` of inferenced feature points.
        '''
        return self.model.forward_propagation(F, x)

    # is-a `mxnet.gluon.hybrid.hybridblock.HybridBlock`.
    __model = None

    def get_model(self):
        ''' getter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        return self.__model
    
    def set_model(self, value):
        ''' setter for `mxnet.gluon.hybrid.hybridblock.HybridBlock`. '''
        self.__model = value
    
    model = property(get_model, set_model)
