# -*- coding: utf-8 -*-
from accelbrainbase.controllablemodel._torch.gan_controller import GANController
from accelbrainbase.observabledata._torch.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.computableloss._torch.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._torch.discriminator_loss import DiscriminatorLoss
import numpy as np
from logging import getLogger
import torch


class GanomalyController(GANController):
    '''
    GANomaly, Semi-Supervised Anomaly Detection via Adversarial Training.

    This is the beta version.

    GANomaly is a model of semi-supervised anomaly detection, which is a 
    novel adversarial autoencoder within an encoder-decoder-encoder pipeline, 
    capturing the training data distribution within both image and latent vector 
    space, yielding superior results to contemporary GAN-based and traditional 
    autoencoder-based approaches.

    **Note** that this model do not follow the models of Akcay, S. et al (2018) 
    due to the specifications of `accel-brain-code`. For instance, this class do not 
    use not LeakyReLu but ReLu.

    References:
        - Akcay, S., Atapour-Abarghouei, A., & Breckon, T. P. (2018, December). Ganomaly: Semi-supervised anomaly detection via adversarial training. In Asian Conference on Computer Vision (pp. 622-637). Springer, Cham.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).

    '''
    # `bool` that means initialization in this class will be deferred or not.
    __init_deferred_flag = False

    def __init__(
        self,
        generative_model,
        re_encoder_model,
        discriminative_model,
        advarsarial_loss,
        encoding_loss,
        contextual_loss,
        discriminator_loss,
        feature_matching_loss=None,
        learning_rate=1e-05,
        ctx="cpu",
        anomaly_score_lambda=0.5,
    ):
        '''
        Init.

        Args:
            generative_model:               is-a `GenerativeModel`.
            re_encoder_model:               is-a `HybridBlock`.
            discriminative_model:           is-a `DiscriminativeModel`.
            advarsarial_loss:               is-a `mxnet.gluon.loss`.
            encoding_loss:                  is-a `mxnet.gluon.loss`.
            contextual_loss:                is-a `mxnet.gluon.loss`.
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
            anomaly_score_lambda:           `float` of trade-off parameter for computing the anomaly scores.
                                            Anomaly score = `anomaly_score_lambda` * `contextual_loss` + (1 - `anomaly_score_lambda`) * `encoding_loss`.

        '''
        if isinstance(generative_model, GenerativeModel) is False:
            raise TypeError("The type of `generative_model` must be `GenerativeModel`.")
        if isinstance(discriminative_model, DiscriminativeModel) is False:
            raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")
        if isinstance(discriminator_loss, DiscriminatorLoss) is False:
            raise TypeError("The type of `discriminator_loss` must be `DiscriminatorLoss`.")

        super().__init__(
            true_sampler=TrueSampler(),
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            discriminator_loss=discriminator_loss,
            generator_loss=GeneratorLoss(),
            feature_matching_loss=feature_matching_loss,
            learning_rate=learning_rate,
            ctx=ctx,
        )

        self.__re_encoder_model = re_encoder_model
        self.__advarsarial_loss = advarsarial_loss
        self.__encoding_loss = encoding_loss
        self.__contextual_loss = contextual_loss

        self.__anomaly_score_lambda = anomaly_score_lambda

        logger = getLogger("accelbrainbase")
        self.__logger = logger

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
            generated_arr, observed_arr, decoded_arr = self.generative_model.draw()

            if hasattr(self.discriminative_model.model, "optimizer") is False:
                _ = self.discriminative_model.inference(observed_arr)

            self.discriminative_model.model.optimizer.zero_grad()
            observed_posterior_arr = self.discriminative_model.inference(observed_arr)
            decoded_posterior_arr = self.discriminative_model.inference(decoded_arr)

            loss = self.discriminator_loss(
                observed_posterior_arr,
                decoded_posterior_arr
            )
            loss.backward()
            self.discriminative_model.model.optimizer.step()

            d_loss += loss
            posterior += decoded_posterior_arr

        d_loss = d_loss / k_step
        posterior = posterior / k_step
        _d_loss = d_loss.to("cpu").detach().numpy().copy()
        _posterior = posterior.to("cpu").detach().numpy().copy()
        return _d_loss, _posterior.mean()

    def train_by_feature_matching(self, k_step):
        feature_matching_loss = 0.0
        return feature_matching_loss

    def train_generator(self):
        '''
        Train generator.
        
        Returns:
            Tuple data.
            - generative loss.
            - discriminative posterior.
        '''

        if hasattr(self.generative_model.model.encoder, "optimizer") is False:
            _ = self.generative_model.draw()

        self.generative_model.model.encoder.optimizer.zero_grad()
        self.generative_model.model.decoder.optimizer.zero_grad()
        generated_arr, observed_arr, decoded_arr = self.generative_model.draw()
        re_encoded_arr = self.__re_encoder_model(decoded_arr)

        observed_posterior_arr = self.discriminative_model.inference(observed_arr)
        decoded_posterior_arr = self.discriminative_model.inference(decoded_arr)

        advarsarial_loss = self.__advarsarial_loss(
            observed_posterior_arr,
            decoded_posterior_arr
        )

        contextual_loss = self.__contextual_loss(
            observed_arr,
            decoded_arr
        )
        encoding_loss = self.__encoding_loss(
            generated_arr,
            re_encoded_arr
        )
        loss = advarsarial_loss + contextual_loss + encoding_loss

        loss.backward()
        self.generative_model.model.decoder.optimizer.step()
        self.generative_model.model.encoder.optimizer.step()

        _loss = loss.to("cpu").detach().numpy()
        _decoded_posterior_arr = decoded_posterior_arr.to("cpu").detach().numpy()
        return _loss, _decoded_posterior_arr.mean()

    def infernce_anomaly_score(self, observed_arr):
        '''
        Infernce the anomaly scores.

        Args:
            observed_arr:       `mxnet.ndarray` of observed data points.
        
        Returns:
            `mxnet.ndarray` of the anomaly scores.
        '''
        if self.generative_model.condition_sampler is None:
            observed_arr = self.generative_model.noise_sampler.draw()
            encoded_arr = self.generative_model.model.encoder(observed_arr)
            decoded_arr = self.generative_model.model.decoder(encoded_arr)
            generated_arr = encoded_arr
        else:
            condition_arr, sampled_arr = self.generative_model.condition_sampler.draw()
            if sampled_arr is not None:
                sampled_arr = sampled_arr + self.generative_model.noise_sampler.draw()
                encoded_arr = self.generative_model.model.encoder(sampled_arr)
                decoded_arr = self.generative_model.model.decoder(encoded_arr)

                observed_arr = sampled_arr

                generated_arr = torch.cat(
                    (
                        encoded_arr,
                        condition_arr,
                    ),
                    dim=self.generative_model.conditonal_dim
                )
            else:
                condition_arr = condition_arr + self.generative_model.noise_sampler.draw()
                encoded_arr = self.generative_model.model.encoder(condition_arr)
                decoded_arr = self.generative_model.model.decoder(encoded_arr)

                observed_arr = condition_arr
                generated_arr = encoded_arr

        re_encoded_arr = self.__re_encoder_model(decoded_arr)
        encoding_loss = self.__encoding_loss(
            generated_arr,
            re_encoded_arr
        )
        contextual_loss = self.__contextual_loss(
            observed_arr,
            decoded_arr
        )

        loss_ = (self.__anomaly_score_lambda * encoding_loss) + ((1 - self.__anomaly_score_lambda) * contextual_loss)
        #_loss = loss_.to("cpu").detach().numpy().copy()
        return loss_
