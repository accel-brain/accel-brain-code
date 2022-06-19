# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel

from accelbrainbase.observabledata._torch.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.samplabledata.true_sampler import TrueSampler

from accelbrainbase.computableloss._torch.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._torch.discriminator_loss import DiscriminatorLoss

import numpy as np
from logging import getLogger

import torch
from torch import nn
from torch.optim.sgd import SGD


class GANController(nn.Module, ControllableModel):
    '''
    The Generative Adversarial Networks(GANs).

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
        discriminator_loss,
        generator_loss=None,
        feature_matching_loss=None,
        discriminator_optimizer_f=None,
        generator_optimizer_f=None,
        learning_rate=1e-05,
        ctx="cpu",
        not_init_flag=False,
    ):
        '''
        Init.

        Args:
            true_sampler:                   is-a `TrueSampler`.
            generative_model:               is-a `GenerativeModel`.
            discriminative_model:           is-a `DiscriminativeModel`.
            generator_loss:                 is-a `GeneratorLoss`.
            discriminator_loss:             is-a `DiscriminatorLoss`.
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
        if isinstance(true_sampler, TrueSampler) is False:
            raise TypeError("The type of `true_sampler` must be `TrueSampler`.")
        if isinstance(generative_model, GenerativeModel) is False:
            raise TypeError("The type of `generative_model` must be `GenerativeModel`.")
        if isinstance(discriminative_model, DiscriminativeModel) is False:
            raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        if generator_loss is None:
            generator_loss = GeneratorLoss(weight=1.0)

        if isinstance(generator_loss, GeneratorLoss) is False:
            raise TypeError("The type of `generator_loss` must be `GeneratorLoss`.")
        if isinstance(discriminator_loss, DiscriminatorLoss) is False:
            raise TypeError("The type of `discriminator_loss` must be `DiscriminatorLoss`.")

        super(GANController, self).__init__()

        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__generator_loss = generator_loss
        self.__discriminator_loss = discriminator_loss
        self.__feature_matching_loss = feature_matching_loss
        self.__learning_rate = learning_rate

        logger = getLogger("accelbrainbase")
        self.__logger = logger
        self.__not_init_flag = not_init_flag

        if self.init_deferred_flag is False:
            if self.__not_init_flag is False:
                if discriminator_optimizer_f is not None:
                    self.discriminator_optimizer = discriminator_optimizer_f(
                        self.__discriminative_model.parameters()
                    )
                else:
                    self.discriminator_optimizer = SGD(
                        self.__discriminative_model.parameters(),
                        lr=self.__learning_rate
                    )

                if generator_optimizer_f is not None:
                    self.generator_optimizer = generator_optimizer_f(
                        self.generative_model.parameters()
                    )
                else:
                    self.generator_optimizer = SGD(
                        self.generative_model.parameters(),
                        lr=self.__learning_rate
                    )

        self.__total_iter_n = 0
        self.__generative_loss_arr = None
        self.__discriminative_loss_arr = None
        self.__posterior_logs_arr = None
        self.__feature_matching_loss_arr = None

    def learn(
        self, 
        iter_n=1000,
        k_step=10,
    ):
        '''
        Learning.

        Args:
            iter_n:                         `int` of the number of training iterations.
            k_step:                         `int` of the number of learning of the `discriminative_model`.
        '''

        g_logs_list = []
        d_logs_list = []
        feature_matching_loss_list = []

        posterior_logs_list = []

        learning_rate = self.__learning_rate

        try:
            for n in range(iter_n):
                if (n + 1) % 100 == 0 or n < 100:
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("Iterations: (" + str(n+1) + "/" + str(iter_n) + ")")
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("The discriminator's turn.")
                    self.__logger.debug("-" * 100)

                loss, posterior = self.train_discriminator(k_step)
                if isinstance(loss, torch.Tensor) is True:
                    _loss = loss.to('cpu').detach().numpy()
                    _posterior = posterior.to('cpu').detach().numpy()
                else:
                    _loss = loss
                    _posterior = posterior

                d_logs_list.append(_loss)
                posterior_logs_list.append(_posterior)

                loss = self.train_by_feature_matching(k_step)
                if isinstance(loss, torch.Tensor) is True:
                    _loss = loss.to('cpu').detach().numpy()
                else:
                    _loss = loss
                feature_matching_loss_list.append(_loss)

                if (n + 1) % 100 == 0 or n < 100:
                    if len(posterior_logs_list) > 0:
                        self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))
                    if len(d_logs_list) > 0:
                        self.__logger.debug("The discriminator's loss(mean): " + str(d_logs_list[-1]))
                    if len(feature_matching_loss_list) > 0:
                        self.__logger.debug("The discriminator's feature matching loss(mean): " + str(feature_matching_loss_list[-1]))

                    self.__logger.debug("-" * 100)
                    self.__logger.debug("The generator's turn.")
                    self.__logger.debug("-" * 100)

                loss, posterior = self.train_generator()
                if isinstance(loss, torch.Tensor) is True:
                    _loss = loss.to('cpu').detach().numpy()
                    _posterior = posterior.to('cpu').detach().numpy()
                else:
                    _loss = loss
                    _posterior = posterior

                g_logs_list.append(_loss)
                posterior_logs_list.append(_posterior)

                if (n + 1) % 100 == 0 or n < 100:
                    if len(g_logs_list) > 0:
                        self.__logger.debug("The generator's loss(mean): " + str(g_logs_list[-1]))
                    if len(posterior_logs_list) > 0:
                        self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))

                self.__total_iter_n = self.__total_iter_n + 1

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

        if self.__generative_loss_arr is None:
            self.__generative_loss_arr = np.array(g_logs_list)
        else:
            self.__generative_loss_arr = np.r_[
                self.__generative_loss_arr,
                np.array(g_logs_list)
            ]

        if self.__discriminative_loss_arr is None:
            self.__discriminative_loss_arr = np.array(d_logs_list)
        else:
            self.__discriminative_loss_arr = np.r_[
                self.__discriminative_loss_arr,
                np.array(d_logs_list)
            ]

        if self.__posterior_logs_arr is None:
            self.__posterior_logs_arr = np.array(posterior_logs_list)
        else:
            self.__posterior_logs_arr = np.r_[
                self.__posterior_logs_arr,
                np.array(posterior_logs_list)
            ]

        if self.__feature_matching_loss_arr is None:
            self.__feature_matching_loss_arr = np.array(feature_matching_loss_list)
        else:
            self.__feature_matching_loss_arr = np.r_[
                self.__feature_matching_loss_arr,
                np.array(feature_matching_loss_list)
            ]

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
            generated_arr = self.__generative_model.draw()

            self.discriminator_optimizer.zero_grad()

            true_posterior_arr = self.__discriminative_model.inference(true_arr)
            generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
            loss = self.__discriminator_loss(
                true_posterior_arr,
                generated_posterior_arr
            )
            loss.backward()
            self.discriminator_optimizer.step()

            d_loss += loss
            posterior += torch.mean(generated_posterior_arr)

        d_loss = d_loss / k_step
        posterior = posterior / k_step
        return d_loss, posterior

    def train_by_feature_matching(self, k_step):
        feature_matching_loss = 0.0
        if self.feature_matching_loss is None:
            return feature_matching_loss

        for k in range(k_step):
            true_arr = self.__true_sampler.draw()
            generated_arr = self.__generative_model.draw()

            self.discriminator_optimizer.zero_grad()

            true_posterior_arr = self.__discriminative_model.inference(true_arr)
            generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
            loss = self.feature_matching_loss(
                true_posterior_arr,
                generated_posterior_arr
            )
            loss.backward()
            self.discriminator_optimizer.step()

            feature_matching_loss += loss

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
        self.generator_optimizer.zero_grad()
        generated_arr = self.__generative_model.draw()
        generated_posterior_arr = self.__discriminative_model.inference(generated_arr)
        loss = self.__generator_loss(generated_posterior_arr)
        loss.backward()
        self.generator_optimizer.step()

        return loss, torch.mean(generated_posterior_arr)

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        torch.save(
            {
                'total_iter_n': self.__total_iter_n,
                'discriminative_model_state_dict': self.discriminative_model.state_dict(),
                'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
                'generative_model_state_dict': self.generative_model.state_dict(),
                'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
                'generative_loss': self.__generative_loss_arr,
                'discriminative_loss': self.__discriminative_loss_arr,
                'posterior_logs': self.__posterior_logs_arr,
                'feature_matching_loss': self.__feature_matching_loss_arr,
            }, 
            filename
        )

    def load_parameters(self, filename, ctx=None, strict=True):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            Context-manager that changes the selected device.
            strict:         Whether to strictly enforce that the keys in state_dict match the keys returned by this module’s state_dict() function. Default: `True`.
        '''
        checkpoint = torch.load(filename)
        self.__total_iter_n = checkpoint["total_iter_n"]
        self.discriminative_model.model.load_state_dict(
            checkpoint["discriminative_model_state_dict"], 
            strict=strict
        )
        self.generative_model.model.load_state_dict(
            checkpoint["generative_model_state_dict"], 
            strict=strict
        )

        self.discriminator_optimizer.load_state_dict(
            checkpoint['discriminator_optimizer_state_dict']
        )
        self.generator_optimizer.load_state_dict(
            checkpoint['generator_optimizer_state_dict']
        )

        self.__generative_loss_arr = checkpoint['generative_loss']
        self.__discriminative_loss_arr = checkpoint['discriminative_loss']
        self.__posterior_logs_arr = checkpoint['posterior_logs']
        self.__feature_matching_loss_arr = checkpoint['feature_matching_loss']

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_true_sampler(self):
        ''' getter for `TrueSampler`. '''
        return self.__true_sampler
    
    def set_true_sampler(self, value):
        ''' setter for `TrueSampler`.'''
        self.__true_sampler = value
    
    true_sampler = property(get_true_sampler, set_true_sampler)

    def get_generative_model(self):
        ''' getter for `GenerativeModel`. '''
        return self.__generative_model

    def set_generative_model(self, value):
        ''' getter for `GenerativeModel`. '''
        self.__generative_model = value

    generative_model = property(get_generative_model, set_generative_model)

    def get_discriminative_model(self):
        ''' getter for `DiscriminativeModel`. '''
        return self.__discriminative_model

    def set_discriminative_model(self, value):
        ''' getter for `DiscriminativeModel`. '''
        self.__discriminative_model = value

    discriminative_model = property(get_discriminative_model, set_discriminative_model)

    def get_feature_matching_loss(self):
        ''' getter for `FeatureMatchingLoss`. '''
        return self.__feature_matching_loss
    
    def set_feature_matching_loss(self, value):
        ''' setter for `FeatureMatchingLoss`. '''
        self.__feature_matching_loss = value
    
    feature_matching_loss = property(get_feature_matching_loss, set_feature_matching_loss)

    def get_generator_loss(self):
        ''' getter for `GeneratorLoss`. '''
        return self.__generator_loss

    def set_generator_loss(self, value):
        ''' setter for `GeneratorLoss`. '''
        self.__generator_loss = value

    generator_loss = property(get_generator_loss, set_generator_loss)

    def get_discriminator_loss(self):
        ''' getter for `DiscriminatorLoss`. '''
        return self.__discriminator_loss

    def set_discriminator_loss(self, value):
        ''' getter for `DiscriminatorLoss`. '''
        self.__discriminator_loss = value

    discriminator_loss = property(get_discriminator_loss, set_discriminator_loss)

    def get_generative_loss_arr(self):
        ''' getter for Generator's losses. '''
        return self.__generative_loss_arr
    
    generative_loss_arr = property(get_generative_loss_arr, set_readonly)

    def get_discriminative_loss_arr(self):
        ''' getter for Generator's losses.'''
        return self.__discriminative_loss_arr
    
    discriminative_loss_arr = property(get_discriminative_loss_arr, set_readonly)

    def get_posterior_logs_arr(self):
        ''' getter for logs of posteriors.'''
        return self.__posterior_logs_arr
    
    posterior_logs_arr = property(get_posterior_logs_arr, set_readonly)

    def get_feature_matching_loss_arr(self):
        ''' getter for logs of posteriors. '''
        return self.__feature_matching_loss_arr
    
    feature_matching_loss_arr = property(get_feature_matching_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter for `bool` that means initialization in this class will be deferred or not.'''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter for `bool` that means initialization in this class will be deferred or not.'''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
