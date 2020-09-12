# -*- coding: utf-8 -*-
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase._mxnet._exception.init_deferred_error import InitDeferredError

from mxnet import gluon
from mxnet import autograd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import MXNetError
from logging import getLogger


class ClusterGANController(GANController):
    '''
    The ClusterGAN.

    This is the beta version.

    References:
        - Ghasedi, K., Wang, X., Deng, C., & Huang, H. (2019). Balanced self-paced learning for generative adversarial clustering network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4391-4400).
    '''
    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    # `int` of learning step of Generator.
    __g_step = 1

    def __init__(
        self,
        generative_model,
        clustering_model,
        discriminative_model,
        discriminator_loss,
        generator_loss,
        consistency_loss,
        feature_matching_loss=None,
        initializer=None,
        optimizer_name="SGD",
        learning_rate=1e-05,
        learning_attenuate_rate=1.0,
        attenuate_epoch=50,
        hybridize_flag=True,
        scale=1.0,
        wd=0.07,
        ctx=mx.cpu(),
        **kwargs
    ):
        '''
        Init.

        Args:
            generative_model:               is-a `GenerativeModel`.
            clustering_model:               is-a `GenerativeModel`.
            discriminative_model:           is-a `DiscriminativeModel`.
            generator_loss:                 is-a `GeneratorLoss`.
            discriminator_loss:             is-a `GANDiscriminatorLoss`.
            consistency_loss:               is-a `Loss`.
            feature_matching_loss:          is-a `GANFeatureMatchingLoss`.
            learning_rate:                  `float` of learning rate.
            learning_attenuate_rate:        `float` of attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                `int` of attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.
                                            Additionally, in relation to regularization,
                                            this class constrains weight matrixes every `attenuate_epoch`.
            optimizer_name:                 `str` of name of optimizer.
            hybridize_flag:                  Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            scale:                          `float` of scaling factor for initial parameters.
            ctx:                            `mx.cpu()` or `mx.gpu()`.

        '''
        logger = getLogger("accelbrainbase")
        self.__logger = logger
        init_deferred_flag = self.init_deferred_flag
        self.init_deferred_flag = True

        if generator_loss is None:
            _generator_loss = GeneratorLoss()
        else:
            _generator_loss = generator_loss

        super().__init__(
            true_sampler=TrueSampler(),
            generative_model=clustering_model,
            discriminative_model=discriminative_model,
            generator_loss=_generator_loss,
            discriminator_loss=discriminator_loss,
            feature_matching_loss=feature_matching_loss,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            learning_attenuate_rate=learning_attenuate_rate,
            attenuate_epoch=attenuate_epoch,
            hybridize_flag=hybridize_flag,
            scale=scale,
            ctx=ctx,
            **kwargs
        )
        self.init_deferred_flag = init_deferred_flag

        self.__generative_model = generative_model
        self.__clustering_model = clustering_model
        self.__discriminative_model = discriminative_model
        self.__discriminator_loss = discriminator_loss
        self.__generator_loss = generator_loss
        self.__consistency_loss = consistency_loss
        self.__feature_matching_loss = feature_matching_loss

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
                self.collect_params().initialize(self.initializer, force_reinit=False, ctx=ctx)
                self.__clustering_model.collect_params().initialize(self.initializer, force_reinit=False, ctx=ctx)
                self.__generative_model.collect_params().initialize(self.initializer, force_reinit=False, ctx=ctx)
                self.__discriminative_model.collect_params().initialize(self.initializer, force_reinit=False, ctx=ctx)

                self.clusterer_trainer = gluon.Trainer(
                    self.__clustering_model.collect_params(), 
                    optimizer_name, 
                    {
                        "learning_rate": learning_rate,
                        "wd": wd
                    }
                )
                self.generator_trainer = gluon.Trainer(
                    self.__generative_model.collect_params(), 
                    optimizer_name, 
                    {
                        "learning_rate": learning_rate,
                        "wd": wd
                    }
                )
                self.discriminator_trainer = gluon.Trainer(
                    self.__discriminative_model.collect_params(),
                    optimizer_name,
                    {
                        "learning_rate": learning_rate,
                        "wd": wd
                    }
                )

                if hybridize_flag is True:
                    self.generative_model.hybridize()
                    self.generative_model.model.hybridize()
                    self.clustering_model.hybridize()
                    self.clustering_model.model.hybridize()

                    if self.clustering_model.model.output_nn is not None:
                        self.clustering_model.model.output_nn.hybridize()

                    self.discriminative_model.hybridize()
                    self.discriminative_model.model.hybridize()

            except InitDeferredError:
                self.__logger.debug("The initialization should be deferred.")

        self.__learning_rate = learning_rate
        self.__learning_attenuate_rate = learning_attenuate_rate
        self.__attenuate_epoch = attenuate_epoch

    def learn(
        self, 
        iter_n=100,
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
        c_logs_list = []
        feature_matching_loss_list = []
        posterior_logs_list = []
        learning_rate = self.__learning_rate

        try:
            for n in range(iter_n):
                if ((n + 1) % self.__attenuate_epoch == 0):
                    learning_rate = learning_rate * self.__learning_attenuate_rate
                    self.clusterer_trainer.set_learning_rate(learning_rate)
                    self.generator_trainer.set_learning_rate(learning_rate)
                    self.discriminator_trainer.set_learning_rate(learning_rate)

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
                self.__logger.debug("The discriminator's loss(mean): " + str(d_logs_list[-1]))
                self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))
                self.__logger.debug("The discriminator's feature matching loss(mean): " + str(feature_matching_loss_list[-1]))

                self.__logger.debug("-" * 100)
                self.__logger.debug("The generator's turn.")
                self.__logger.debug("-" * 100)
                loss, posterior = self.train_generator()
                g_logs_list.append(loss)
                posterior_logs_list.append(posterior)
                self.__logger.debug("The generator's loss(mean): " + str(g_logs_list[-1]))
                self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))

                self.__logger.debug("-" * 100)
                self.__logger.debug("The clusterer's turn.")
                self.__logger.debug("-" * 100)
                loss, posterior = self.train_clusterer()
                c_logs_list.append(loss)
                posterior_logs_list.append(posterior)
                self.__logger.debug("The clusterer's loss(mean): " + str(c_logs_list[-1]))
                self.__logger.debug("The discriminator's posterior(mean): " + str(posterior_logs_list[-1]))

        except KeyboardInterrupt:
            print("Keyboard Interrupt.")

        self.__discriminative_loss_arr = np.array(d_logs_list)
        self.__generative_loss_arr = np.array(g_logs_list)
        self.__clusterer_loss_arr = np.array(c_logs_list)
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
            with autograd.record():
                with autograd.predict_mode():
                    generated_arr, true_soft_assignment_arr = self.__generative_model.draw()
                    clustered_arr, soft_assignment_arr = self.clustering_model.draw()

                self.__generated_shape = clustered_arr.shape
                generated_arr = generated_arr.reshape(self.__generated_shape)

                x = (generated_arr, true_soft_assignment_arr)
                g_feature_arr = None
                for i in range(len(self.discriminative_model.model_list)):
                    arr = self.discriminative_model.model_list[i](x[i])
                    if g_feature_arr is None:
                        g_feature_arr = arr
                    else:
                        g_feature_arr = nd.concat(g_feature_arr, arr, dim=1)

                generated_posterior_arr = self.discriminative_model.inference(g_feature_arr)
                x = (clustered_arr, soft_assignment_arr)
                c_feature_arr = None
                for i in range(len(self.discriminative_model.model_list)):
                    arr = self.discriminative_model.model_list[i](x[i])
                    if c_feature_arr is None:
                        c_feature_arr = arr
                    else:
                        c_feature_arr = nd.concat(c_feature_arr, arr, dim=1)

                clustered_posterior_arr = self.discriminative_model.inference(c_feature_arr)
                loss = self.discriminator_loss(
                    generated_posterior_arr,
                    clustered_posterior_arr
                )

            loss.backward()
            self.discriminator_trainer.step(generated_arr.shape[0])
            self.discriminative_model.model.regularize()
            [self.discriminative_model.model_list[i].regularize() for i in range(len(self.discriminative_model.model_list))]
            d_loss += loss.mean().asnumpy()[0]
            posterior += generated_posterior_arr.mean().asnumpy()[0]

        d_loss = d_loss / k_step
        posterior = posterior / k_step

        return d_loss, posterior

    def train_generator(self):
        '''
        Train clusterer and generator.
        
        Returns:
            Tuple data.
            - generative loss.
            - discriminative posterior.
        '''
        total_g_loss = 0.0
        total_posterior = 0.0
        for g in range(self.g_step):
            with autograd.record():
                generated_arr, true_soft_assignment_arr = self.__generative_model.draw()
                generated_arr = generated_arr.reshape(self.__generated_shape)

                with autograd.predict_mode():
                    #predictive_model_flag = self.clustering_model.predictive_model_flag
                    #self.clustering_model.predictive_model_flag = True
                    clustered_arr, soft_assignment_arr = self.clustering_model.draw()
                    #self.clustering_model.predictive_model_flag = predictive_model_flag

                    x = (generated_arr, true_soft_assignment_arr)
                    g_feature_arr = None
                    for i in range(len(self.discriminative_model.model_list)):
                        arr = self.discriminative_model.model_list[i](x[i])
                        if g_feature_arr is None:
                            g_feature_arr = arr
                        else:
                            g_feature_arr = nd.concat(g_feature_arr, arr, dim=1)

                    generated_posterior_arr = self.__discriminative_model.inference(g_feature_arr)

                    if isinstance(self.discriminator_loss, DiscriminatorLoss) is False:
                        if self.__generator_loss is not None:
                            g_loss = self.__generator_loss(generated_posterior_arr)
                        else:
                            g_loss = 0.0
                    else:
                        g_loss = generated_posterior_arr

                c_loss = self.__consistency_loss(clustered_arr, generated_arr)

                loss = g_loss + c_loss
            loss.backward()

            self.generator_trainer.step(generated_arr.shape[0])
            self.__generative_model.model.regularize()

            total_g_loss += loss.mean().asnumpy()[0]
            total_posterior += generated_posterior_arr.mean().asnumpy()[0]

        total_g_loss = total_g_loss / self.g_step
        total_posterior = total_posterior / self.g_step

        return total_g_loss, total_posterior

    def train_clusterer(self):
        '''
        Train clusterer.
        
        Returns:
            Tuple data.
            - generative loss.
            - discriminative posterior.
        '''
        total_g_loss = 0.0
        total_posterior = 0.0
        for g in range(self.g_step):
            with autograd.record():
                with autograd.predict_mode():
                    generated_arr, true_soft_assignment_arr = self.__generative_model.draw()
                    generated_arr = generated_arr.reshape(self.__generated_shape)

                #predictive_model_flag = self.clustering_model.predictive_model_flag
                #self.clustering_model.predictive_model_flag = True
                clustered_arr, soft_assignment_arr = self.clustering_model.draw()
                #self.clustering_model.predictive_model_flag = predictive_model_flag

                with autograd.predict_mode():
                    x = (generated_arr, true_soft_assignment_arr)
                    g_feature_arr = None
                    for i in range(len(self.discriminative_model.model_list)):
                        arr = self.discriminative_model.model_list[i](x[i])
                        if g_feature_arr is None:
                            g_feature_arr = arr
                        else:
                            g_feature_arr = nd.concat(g_feature_arr, arr, dim=1)

                    generated_posterior_arr = self.__discriminative_model.inference(g_feature_arr)

                    if isinstance(self.discriminator_loss, DiscriminatorLoss) is False:
                        if self.__generator_loss is not None:
                            g_loss = self.__generator_loss(generated_posterior_arr)
                        else:
                            g_loss = 0.0
                    else:
                        g_loss = generated_posterior_arr

                c_loss = self.__consistency_loss(clustered_arr, generated_arr)

                loss = g_loss + c_loss
            loss.backward()

            self.clustering_model.model.regularize()

            total_g_loss += loss.mean().asnumpy()[0]
            total_posterior += generated_posterior_arr.mean().asnumpy()[0]

        total_g_loss = total_g_loss / self.g_step
        total_posterior = total_posterior / self.g_step

        return total_g_loss, total_posterior

    def train_by_feature_matching(self, k_step):
        feature_matching_loss = 0.0
        if self.feature_matching_loss is None:
            return feature_matching_loss

        for k in range(k_step):
            with autograd.predict_mode():
                generated_arr, true_soft_assignment_arr = self.__generative_model.draw()
                clustered_arr, soft_assignment_arr = self.clustering_model.draw()
                generated_arr = generated_arr.reshape(self.__generated_shape)

            with autograd.record():
                x = (generated_arr, true_soft_assignment_arr)
                g_feature_arr = None
                for i in range(len(self.discriminative_model.model_list)):
                    arr = self.discriminative_model.model_list[i](x[i])
                    if g_feature_arr is None:
                        g_feature_arr = arr
                    else:
                        g_feature_arr = nd.concat(g_feature_arr, arr, dim=1)

                x = (clustered_arr, soft_assignment_arr)
                c_feature_arr = None
                for i in range(len(self.discriminative_model.model_list)):
                    arr = self.discriminative_model.model_list[i](x[i])
                    if c_feature_arr is None:
                        c_feature_arr = arr
                    else:
                        c_feature_arr = nd.concat(c_feature_arr, arr, dim=1)

                generated_posterior_arr = self.discriminative_model.inference(g_feature_arr)
                clustered_posterior_arr = self.discriminative_model.inference(c_feature_arr)
                loss = self.feature_matching_loss(
                    clustered_posterior_arr,
                    generated_posterior_arr
                )

            loss.backward()
            self.discriminator_trainer.step(generated_posterior_arr.shape[0])
            feature_matching_loss += loss.mean().asnumpy()[0]

        feature_matching_loss = feature_matching_loss / k_step
        return feature_matching_loss

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
        c_filename = filename.replace("." + _format, "_clusterer." + _format)
        g_filename = filename.replace("." + _format, "_generator." + _format)
        d_filename = filename.replace("." + _format, "_discriminator." + _format)
        return c_filename, g_filename, d_filename

    def save_parameters(self, filename):
        '''
        Save parameters to files.

        Args:
            filename:       File name.
        '''
        c_filename, g_filename, d_filename = self.__rename_file(filename)
        self.clustering_model.save_parameters(c_filename)
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
        c_filename, g_filename, d_filename = self.__rename_file(filename)
        self.generative_model.load_parameters(g_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.clustering_model.load_parameters(c_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        self.discriminative_model.load_parameters(d_filename, ctx=ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

    def get_g_step(self):
        ''' getter '''
        return self.__g_step
    
    def set_g_step(self, value):
        ''' setter '''
        self.__g_step = value
    
    g_step = property(get_g_step, set_g_step)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def get_generative_model(self):
        ''' getter '''
        return self.__generative_model
    
    def set_generative_model(self, value):
        ''' setter '''
        self.__generative_model = value
    
    generative_model = property(get_generative_model, set_generative_model)

    def get_clustering_model(self):
        ''' getter '''
        return self.__clustering_model

    def set_clustering_model(self, value):
        ''' getter '''
        self.__clustering_model = value

    clustering_model = property(get_clustering_model, set_clustering_model)

    def get_discriminative_model(self):
        ''' getter '''
        return self.__discriminative_model

    def set_discriminative_model(self, value):
        ''' getter '''
        self.__discriminative_model = value

    discriminative_model = property(get_discriminative_model, set_discriminative_model)

    def get_generative_loss_arr(self):
        ''' getter '''
        return self.__generative_loss_arr
    
    generative_loss_arr = property(get_generative_loss_arr, set_readonly)

    def get_discriminative_loss_arr(self):
        ''' getter '''
        return self.__discriminative_loss_arr
    
    discriminative_loss_arr = property(get_discriminative_loss_arr, set_readonly)

    def get_generative_loss_arr(self):
        ''' getter '''
        return self.__generative_loss_arr
    
    generative_loss_arr = property(get_generative_loss_arr, set_readonly)

    def get_clusterer_loss_arr(self):
        ''' getter '''
        return self.__clusterer_loss_arr
    
    clusterer_loss_arr = property(get_clusterer_loss_arr, set_readonly)

    def get_posterior_logs_arr(self):
        ''' getter '''
        return self.__posterior_logs_arr

    posterior_logs_arr = property(get_posterior_logs_arr, set_readonly)

    def get_feature_matching_loss_arr(self):
        ''' getter '''
        return self.__feature_matching_loss_arr
    
    feature_matching_loss_arr = property(get_feature_matching_loss_arr, set_readonly)

    def get_init_deferred_flag(self):
        ''' getter '''
        return self.__init_deferred_flag
    
    def set_init_deferred_flag(self, value):
        ''' setter '''
        self.__init_deferred_flag = value

    init_deferred_flag = property(get_init_deferred_flag, set_init_deferred_flag)
