# -*- coding: utf-8 -*-
from accelbrainbase.computableloss._torch.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata._torch.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata._torch.unlabeled_image_iterator import UnlabeledImageIterator
from accelbrainbase.noiseabledata._torch.gauss_noise import GaussNoise
from accelbrainbase.observabledata._torch.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._torch.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._torch.convolutionalneuralnetworks.convolutionalautoencoder.contractive_cae import ContractiveCAE as ConvolutionalAutoEncoder
from accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._torch.adversarialmodel.discriminativemodel.eb_discriminative_model import EBDiscriminativeModel
from accelbrainbase.observabledata._torch.adversarialmodel.generativemodel.convolutional_auto_encoder import ConvolutionalAutoEncoder as GenerativeModel
from accelbrainbase.computableloss._torch.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._torch.discriminator_loss import DiscriminatorLoss
from accelbrainbase.computableloss._torch.discriminatorloss.eb_discriminator_loss import EBDiscriminatorLoss
from accelbrainbase.observabledata._torch.adversarialmodel.discriminativemodel.eb_discriminative_model import EBDiscriminativeModel

from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.truesampler._torch.normal_true_sampler import NormalTrueSampler

from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._torch.uniform_noise_sampler import UniformNoiseSampler
from accelbrainbase.controllablemodel._torch.gancontroller.aae_controller import AAEController
from accelbrainbase.controllablemodel._torch.gancontroller.aaecontroller.ebaae_controller import EBAAEController

import numpy as np
import pandas as pd

import torch


class EBAAEImageGenerator(object):
    '''
    Image generation by EBAAE.

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

    On the other hand, models that construct a discriminator by Auto-Encoder have been proposed. 
    The Energy-based GAN(EBGAN) framework considers the discriminator as an energy function, 
    which assigns low energy values to real data and high to fake data. The generator is a 
    trainable parameterized function that produces samples in regions to which the discriminator 
    assigns low energy.

    This class models the Energy-based Adversarial-Auto-Encoder(EBAAE) by structural coupling 
    between AAEs and EBGAN. The learning algorithm equivalents an adversarial training of AAEs as 
    a generator and EBGAN as a discriminator.

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
        dir_list,
        width=28,
        height=28,
        channel=1,
        initializer_f=None,
        optimizer_f=None,
        normal_height=14,
        normal_width=14,
        normal_channel=32,
        batch_size=40,
        learning_rate=1e-03,
        ctx="cpu",
        discriminative_model=None,
        generative_model=None,
        discriminator_loss_weight=1.0,
        reconstruction_loss_weight=1.0,
        feature_matching_loss_weight=1.0,
    ):
        '''
        Init.

        If you are not satisfied with this simple default setting,
        delegate `discriminative_model` and `generative_model` designed by yourself.

        Args:
            dir_list:       `list` of `str` of path to image files.
            width:          `int` of image width.
            height:         `int` of image height.
            channel:        `int` of image channel.

            normal_width:   `int` of width of image drawn from normal distribution, p(z).
            normal_height:  `int` of height of image drawn from normal distribution, p(z).
            normal_channel: `int` of channel of image drawn from normal distribution, p(z).

            initializer:    is-a `mxnet.initializer` for parameters of model.
                            If `None`, it is drawing from the Xavier distribution.
            
            batch_size:     `int` of batch size.
            learning_rate:  `float` of learning rate.
            ctx:            `mx.gpu()` or `mx.cpu()`.

            discriminative_model:       is-a `accelbrainbase.observabledata._torch.adversarialmodel.discriminative_model.discriminativemodel.eb_discriminative_model.EBDiscriminativeModel`.
            generative_model:           is-a `accelbrainbase.observabledata._torch.adversarialmodel.generative_model.GenerativeModel`.

            discriminator_loss_weight:      `float` of weight for discriminator loss.
            reconstruction_loss_weight:     `float` of weight for reconstruction loss.
            feature_matching_loss_weight:   `float` of weight for feature matching loss.
        '''
        image_extractor = ImageExtractor(
            width=width,
            height=height,
            channel=channel,
            ctx=ctx
        )

        unlabeled_image_iterator = UnlabeledImageIterator(
            image_extractor=image_extractor,
            dir_list=dir_list,
            batch_size=batch_size,
            norm_mode="z_score",
            scale=1.0,
            noiseable_data=GaussNoise(sigma=1e-03, mu=0.0),
        )

        computable_loss = L2NormLoss()

        if discriminative_model is None:
            d_encoder = ConvolutionalNeuralNetworks(
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `list` of int` of the number of units in hidden layers.
                hidden_units_list=[
                    torch.nn.Conv2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ), 
                    torch.nn.Conv2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ],
                # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                hidden_activation_list=[
                    torch.nn.ReLU(), 
                    torch.nn.ReLU()
                ],
                # `list` of `float` of dropout rate.
                hidden_dropout_rate_list=[0.5, 0.5],
                # `list` of `mxnet.gluon.nn.BatchNorm`.
                hidden_batch_norm_list=[
                    torch.nn.BatchNorm2d(normal_channel), 
                    torch.nn.BatchNorm2d(normal_channel)
                ],
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )

            d_decoder = ConvolutionalNeuralNetworks(
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `list` of int` of the number of units in hidden layers.
                hidden_units_list=[
                    torch.nn.ConvTranspose2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ), 
                    torch.nn.ConvTranspose2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ],
                # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                hidden_activation_list=[
                    torch.nn.ReLU(), 
                    "identity"
                ],
                # `list` of `float` of dropout rate.
                hidden_dropout_rate_list=[0.5, 0.0],
                # `list` of `mxnet.gluon.nn.BatchNorm`.
                hidden_batch_norm_list=[
                    torch.nn.BatchNorm2d(normal_channel), 
                    None
                ],
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )
            d_model = ConvolutionalAutoEncoder(
                # is-a `ConvolutionalNeuralNetworks`.
                encoder=d_encoder,
                # is-a `ConvolutionalNeuralNetworks`.
                decoder=d_decoder,
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `bool` of flag to tied weights or not.
                tied_weights_flag=True,
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )
            d_model.batch_size = batch_size

            discriminative_model = EBDiscriminativeModel(
                # is-a `ConvolutionalAutoEncoder`.
                model=d_model, 
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx, 
            )
        else:
            if isinstance(discriminative_model, DiscriminativeModel) is False:
                raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        if generative_model is None:
            encoder = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
                learning_rate=learning_rate,
                hidden_units_list=[
                    torch.nn.Conv2d(
                        in_channels=channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ), 
                    torch.nn.Conv2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                hidden_dropout_rate_list=[0.5, 0.5],
                hidden_batch_norm_list=[
                    torch.nn.BatchNorm2d(normal_channel), 
                    torch.nn.BatchNorm2d(normal_channel)
                ],
                hidden_activation_list=[
                    torch.nn.ReLU(), 
                    torch.nn.ReLU()
                ],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                regularizatable_data_list=[],
            )

            decoder = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer_f=initializer_f,
                optimizer_f=optimizer_f,
                learning_rate=learning_rate,
                hidden_units_list=[
                    torch.nn.ConvTranspose2d(
                        in_channels=normal_channel,
                        out_channels=normal_channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ), 
                    torch.nn.ConvTranspose2d(
                        in_channels=normal_channel,
                        out_channels=channel,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                hidden_dropout_rate_list=[0.5, 0.0],
                hidden_batch_norm_list=[
                    torch.nn.BatchNorm2d(normal_channel), 
                    None
                ],
                hidden_activation_list=[
                    torch.nn.ReLU(), 
                    "identity"
                ],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                regularizatable_data_list=[],
            )

            g_model = ConvolutionalAutoEncoder(
                encoder=encoder,
                decoder=decoder,
                computable_loss=computable_loss,
                learning_rate=learning_rate,
                ctx=ctx,
                regularizatable_data_list=[],
            )
            d_model.batch_size = 40

            true_sampler = TrueSampler()
            true_sampler.iteratorable_data = unlabeled_image_iterator

            condition_sampler = ConditionSampler()
            condition_sampler.true_sampler = true_sampler

            generative_model = GenerativeModel(
                noise_sampler=UniformNoiseSampler(
                    low=-1e-03,
                    high=1e-03,
                    batch_size=batch_size,
                    seq_len=0,
                    channel=channel,
                    height=height,
                    width=width,
                    ctx=ctx
                ), 
                model=g_model, 
                condition_sampler=condition_sampler,
                conditonal_dim=1,
                learning_rate=learning_rate,
                ctx=ctx, 
            )
        else:
            if isinstance(generative_model, GenerativeModel) is False:
                raise TypeError("The type of `generative_model` must be `GenerativeModel`.")

        normal_ture_sampler = NormalTrueSampler(
            batch_size=batch_size,
            seq_len=0,
            channel=normal_channel,
            height=normal_height,
            width=normal_width,
            ctx=ctx
        )

        EBAAE = EBAAEController(
            true_sampler=normal_ture_sampler,
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            discriminator_loss=EBDiscriminatorLoss(weight=discriminator_loss_weight),
            reconstruction_loss=L2NormLoss(weight=reconstruction_loss_weight),
            feature_matching_loss=L2NormLoss(weight=feature_matching_loss_weight),
            learning_rate=learning_rate,
            ctx=ctx,
        )
        self.EBAAE = EBAAE

    def learn(self, iter_n=1000, k_step=10):
        '''
        Learning.

        Args:
            iter_n:                         `int` of the number of training iterations.
            k_step:                         `int` of the number of learning of the `discriminative_model`.
        '''
        self.EBAAE.learn(
            iter_n=iter_n,
            k_step=k_step,
        )
