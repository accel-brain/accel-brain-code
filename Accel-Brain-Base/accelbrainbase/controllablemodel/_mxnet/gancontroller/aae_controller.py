# -*- coding: utf-8 -*-
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.samplabledata.true_sampler import TrueSampler

from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss

from mxnet.gluon.block import HybridBlock
from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
from mxnet import MXNetError
from logging import getLogger


class AAEController(GANController):
    '''
    The Adversarial Auto-Encoder(AAE).

    The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes 
    a min-max adversarial game between two neural networks – a generative model, `G`, and a 
    discriminative model, `D`. The discriminator model, `D(x)`, is a neural network that computes 
    the probability that a observed data point `x` in data space is a sample from the data 
    distribution (positive samples) that we are trying to model, rather than a sample from our 
    generative model (negative samples). 
    
    Concurrently, the generator uses a function `G(z)` that maps samples `z` from the prior `p(z)` to 
    the data space. `G(z)` is trained to maximally confuse the discriminator into believing that 
    samples it generates come from the data distribution. The generator is trained by leveraging 
    the gradient of `D(x)` w.r.t. x, and using that to modify its parameters.

    The Conditional GANs (or cGANs) is a simple extension of the basic GAN model which allows 
    the model to condition on external information. This makes it possible to engage the learned 
    generative model in different "modes" by providing it with different contextual 
    information (Gauthier, J. 2014).

    This model can be constructed by simply feeding the data, `y`, to condition on to both the generator 
    and discriminator. In an unconditioned generative model, because the maps samples `z` from the prior 
    `p(z)` are drawn from uniform or normal distribution, there is no control on modes of the data being 
    generated. On the other hand, it is possible to direct the data generation process by conditioning 
    the model on additional information (Mirza, M., & Osindero, S. 2014).

    This library also provides the Adversarial Auto-Encoders(AAEs), 
    which is a probabilistic Auto-Encoder that uses GANs to perform variational 
    inference by matching the aggregated posterior of the feature points 
    in hidden layer of the Auto-Encoder with an arbitrary prior 
    distribution(Makhzani, A., et al., 2015). Matching the aggregated posterior 
    to the prior ensures that generating from any part of prior space results 
    in meaningful samples. As a result, the decoder of the Adversarial Auto-Encoder 
    learns a deep generative model that maps the imposed prior to the data distribution.

    References:
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
        - Mirza, M., & Osindero, S. (2014). Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784.
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).
        - Zhao, J., Mathieu, M., & LeCun, Y. (2016). Energy-based generative adversarial network. arXiv preprint arXiv:1609.03126.
        - Warde-Farley, D., & Bengio, Y. (2016). Improving generative adversarial networks with denoising feature matching.

    '''
    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        true_sampler,
        generative_model,
        discriminative_model,
        generator_loss,
        discriminator_loss,
        reconstruction_loss,
        feature_matching_loss=None,
        optimizer_name="SGD",
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        hybridize_flag=True,
        scale=1.0,
        ctx=mx.gpu(),
        initializer=None,
        **kwargs
    ):
        '''
        Init.

        Args:
            true_sampler:                   is-a `TrueSampler` as a prior distribution `P(z)`.
            generative_model:               is-a `GenerativeModel`.
            discriminative_model:           is-a `DiscriminativeModel`.
            generator_loss:                 is-a `GeneratorLoss`.
            discriminator_loss:             is-a `DiscriminatorLoss`.
            reconstruction_loss:            is-a `mxnet.gluon.loss` to compute reconstruction loss of Auto-Encoder.
            feature_matching_loss:          is-a `mxnet.gluon.loss`.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.

        '''
        super().__init__(
            true_sampler=true_sampler,
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            feature_matching_loss=feature_matching_loss,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            hybridize_flag=hybridize_flag,
            scale=scale,
            ctx=ctx,
            initializer=initializer,
            **kwargs
        )

        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__generator_loss = generator_loss
        self.__discriminator_loss = discriminator_loss
        self.__feature_matching_loss = feature_matching_loss
        self.__reconstruction_loss = reconstruction_loss

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

    def train_discriminator(
        self, 
        k_step, 
    ):
        '''
        Training for discriminator.

        Args:
            k_step:                         `int` of the number of learning of the `discriminative_model`.
        
        Returns:
            Tuple data.
            - discriminative loss.
            - discriminative posterior.
        '''
        d_loss = 0.0
        posterior = 0.0
        for k in range(k_step):
            true_arr = self.__true_sampler.draw()
            generated_arr, _, _ = self.__generative_model.draw()

            with autograd.record():
                true_posterior_arr = self.__discriminative_model.inference(true_arr)
                generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
                loss = self.__discriminator_loss(
                    true_posterior_arr,
                    generated_posterior_arr
                )
            loss.backward()
            self.discriminator_trainer.step(true_arr.shape[0])

            d_loss += loss.mean().asnumpy()[0]
            posterior += generated_posterior_arr.mean().asnumpy()[0]

        d_loss = d_loss / k_step
        posterior = posterior / k_step
        return d_loss, posterior

    def train_by_feature_matching(self, k_step):
        feature_matching_loss = 0.0
        if self.feature_matching_loss is None:
            return feature_matching_loss

        for k in range(k_step):
            true_arr = self.__true_sampler.draw()
            generated_arr, _, _ = self.__generative_model.draw()

            with autograd.record():
                true_posterior_arr = self.__discriminative_model.inference(true_arr)
                generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
                loss = self.feature_matching_loss(
                    true_posterior_arr,
                    generated_posterior_arr
                )
            loss.backward()
            self.discriminator_trainer.step(true_arr.shape[0])

            feature_matching_loss += loss.mean().asnumpy()[0]

        feature_matching_loss = feature_matching_loss / k_step
        return feature_matching_loss

    def train_generator(self):
        '''
        Train generator.
        
        Returns:
            Tuple data.
            - generative loss.
            - discriminative posterior.
        '''
        with autograd.record():
            generated_arr, observed_arr, decoded_arr = self.__generative_model.draw()
            with autograd.predict_mode():
                generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
            g_loss = self.__generator_loss(generated_posterior_arr)
            a_loss = self.__reconstruction_loss(observed_arr, decoded_arr)
            loss = g_loss + a_loss
        loss.backward()
        self.generator_trainer.step(generated_arr.shape[0])

        return loss.mean().asnumpy()[0], generated_posterior_arr.mean().asnumpy()[0]
