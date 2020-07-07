# -*- coding: utf-8 -*-
from accelbrainbase.controllable_model import ControllableModel
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


class GANController(HybridBlock, ControllableModel):
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
        generator_loss,
        discriminator_loss,
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
        if isinstance(generator_loss, GeneratorLoss) is False:
            raise TypeError("The type of `generator_loss` must be `GeneratorLoss`.")
        if isinstance(discriminator_loss, DiscriminatorLoss) is False:
            raise TypeError("The type of `discriminator_loss` must be `DiscriminatorLoss`.")

        super(GANController, self).__init__(**kwargs)

        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__generator_loss = generator_loss
        self.__discriminator_loss = discriminator_loss
        self.__feature_matching_loss = feature_matching_loss

        logger = getLogger("accelbrainbase")
        self.__logger = logger

        if initializer is None:
            self.initializer = mx.initializer.Xavier(
                rnd_type="gaussian", 
                factor_type="in", 
                magnitude=1
            )
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")
            self.initializer = initializer

        if self.init_deferred_flag is False:
            try:
                self.collect_params().initialize(self.initializer, force_reinit=True, ctx=ctx)
                self.generator_trainer = gluon.Trainer(
                    self.__generative_model.collect_params(), 
                    optimizer_name, 
                    {
                        "learning_rate": learning_rate
                    }
                )
                self.discriminator_trainer = gluon.Trainer(
                    self.__discriminative_model.collect_params(),
                    optimizer_name,
                    {
                        "learning_rate": learning_rate
                    }
                )
                if hybridize_flag is True:
                    self.__generative_model.hybridize()
                    self.__generative_model.model.hybridize()
                    self.__discriminative_model.hybridize()
                    self.__discriminative_model.model.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

    def collect_params(self, select=None):
        '''
        Overrided `collect_params` in `mxnet.gluon.HybridBlok`.
        '''
        params_dict = self.__generative_model.collect_params(select)
        params_dict.update(self.__discriminative_model.collect_params(select))
        return params_dict

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
                if ((n + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.generator_trainer.set_learning_rate(learning_rate)
                    self.discriminator_trainer.set_learning_rate(learning_rate)

                if (n + 1) % 100 == 0:
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("Iterations: (" + str(n+1) + "/" + str(iter_n) + ")")
                    self.__logger.debug("-" * 100)
                    self.__logger.debug("The discriminator's turn.")
                    self.__logger.debug("-" * 100)

                loss, posterior = self.train_discriminator(k_step)
                d_logs_list.append(loss)
                posterior_logs_list.append(posterior)

                loss = self.train_by_feature_matching(k_step)
                feature_matching_loss_list.append(loss)

                if (n + 1) % 100 == 0:
                    self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))
                    self.__logger.debug("The discriminator's loss(mean): " + str(d_logs_list[-1]))
                    self.__logger.debug("The discriminator's feature matching loss(mean): " + str(feature_matching_loss_list[-1]))

                    self.__logger.debug("-" * 100)
                    self.__logger.debug("The generator's turn.")
                    self.__logger.debug("-" * 100)

                loss, posterior = self.train_generator()
                g_logs_list.append(loss)
                posterior_logs_list.append(posterior)

                if (n + 1) % 100 == 0:
                    self.__logger.debug("The generator's loss(mean): " + str(g_logs_list[-1]))
                    self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

        self.__generative_loss_arr = np.array(g_logs_list)
        self.__discriminative_loss_arr = np.array(d_logs_list)
        self.__posterior_logs_arr = np.array(posterior_logs_list)
        self.__feature_matching_loss_arr = np.array(feature_matching_loss_list)

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
            generated_arr = self.__generative_model.draw()

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
            generated_arr = self.__generative_model.draw()
            with autograd.predict_mode():
                generated_posterior_arr = self.__discriminative_model.inference(generated_arr)

            loss = self.__generator_loss(generated_posterior_arr)

        loss.backward()
        self.generator_trainer.step(generated_arr.shape[0])

        return loss.mean().asnumpy()[0], generated_posterior_arr.mean().asnumpy()[0]

    def extract_learned_dict(self):
        '''
        Extract (pre-) learned parameters.

        Returns:
            `dict` of the parameters.
        '''
        params_dict = self.collect_params()
        
        params_arr_dict = {}
        for k in params_dict:
            params_arr_dict.setdefault(k, params_dict[k].data())

        return params_arr_dict

    def __rename_file(self, filename):
        filename_list = filename.split(".")
        _format = filename_list[-1]
        g_filename = filename.replace("." + _format, "_generator." + _format)
        d_filename = filename.replace("." + _format, "_discriminator." + _format)
        return g_filename, d_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        g_filename, d_filename = self.__rename_file(filename)
        self.generative_model.save_parameters(g_filename)
        self.discriminative_model.save_parameters(d_filename)

    def load_parameters(self, filename, ctx=None, allow_missing=False, ignore_extra=False):
        '''
        Load parameters to files.

        Args:
            filename:       File name.
            ctx:            `mx.cpu()` or `mx.gpu()`.
            allow_missing:  `bool` of whether to silently skip loading parameters not represents in the file.
            ignore_extra:   `bool` of whether to silently ignre parameters from the file that are not present in this `Block`.
        '''
        g_filename, d_filename = self.__rename_file(filename)
        self.generative_model.load_parameters(g_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.discriminative_model.load_parameters(d_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

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
