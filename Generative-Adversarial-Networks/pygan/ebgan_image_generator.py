# -*- coding: utf-8 -*-
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata._mxnet.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_image_iterator import UnlabeledImageIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutionalautoencoder.contractive_cae import ContractiveCAE as ConvolutionalAutoEncoder
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminativemodel.eb_discriminative_model import EBDiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminatorloss.eb_discriminator_loss import EBDiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler
from accelbrainbase.controllablemodel._mxnet.gancontroller.ebgan_controller import EBGANController

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm


class EBGANImageGenerator(object):
    '''
    Image generation by EBGANs.

    The Generative Adversarial Networks(GANs) (Goodfellow et al., 2014) framework establishes 
    a min-max adversarial game between two neural networks – a generative model, `G`, and a 
    discriminative model, `D`. The discriminator model, `D(x)`, is a neural network that computes 
    the probability that a observed data point x in data space is a sample from the data 
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

    The Energy-based GAN framework considers the discriminator as an energy function, 
    which assigns low energy values to real data and high to fake data. 
    The generator is a trainable parameterized function that produces 
    samples in regions to which the discriminator assigns low energy. 

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
        initializer=None,
        batch_size=40,
        learning_rate=1e-03,
        ctx=mx.gpu(),
        discriminative_model=None,
        generative_model=None,
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

            initializer:    is-a `mxnet.initializer` for parameters of model.
                            If `None`, it is drawing from the Xavier distribution.
            
            batch_size:     `int` of batch size.
            learning_rate:  `float` of learning rate.
            ctx:            `mx.gpu()` or `mx.cpu()`.

            discriminative_model:       is-a `accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model.discriminativemodel.eb_discriminative_model.EBDiscriminativeModel`.
            generative_model:           is-a `accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model.GenerativeModel`.

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
            norm_mode="min_max",
            scale=0.9,
            noiseable_data=GaussNoise(sigma=1e-08, mu=0.005),
        )

        true_sampler = TrueSampler()
        true_sampler.iteratorable_data = unlabeled_image_iterator

        computable_loss = L2NormLoss()

        if discriminative_model is None:
            encoder = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                hidden_units_list=[
                    Conv2D(
                        channels=16,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ), 
                    Conv2D(
                        channels=32,
                        kernel_size=3,
                        strides=(1, 1),
                        padding=(1, 1),
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                hidden_dropout_rate_list=[0.5, 0.5],
                hidden_batch_norm_list=[BatchNorm(), BatchNorm()],
                optimizer_name="SGD",
                hidden_activation_list=["relu", "relu"],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

            decoder = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                hidden_units_list=[
                    Conv2DTranspose(
                        channels=16,
                        kernel_size=3,
                        strides=(1, 1),
                        padding=(1, 1),
                    ), 
                    Conv2DTranspose(
                        channels=channel,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                hidden_dropout_rate_list=[0.0, 0.0],
                hidden_batch_norm_list=[BatchNorm(), None],
                optimizer_name="SGD",
                hidden_activation_list=["identity", "sigmoid"],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

            d_model = ConvolutionalAutoEncoder(
                encoder=encoder,
                decoder=decoder,
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                optimizer_name="SGD",
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )
            d_model.batch_size = batch_size

            discriminative_model = EBDiscriminativeModel(
                model=d_model, 
                initializer=initializer,
                learning_rate=learning_rate,
                optimizer_name="SGD",
                hybridize_flag=True,
                scale=1.0, 
                ctx=ctx, 
            )
        else:
            if isinstance(discriminative_model, EBDiscriminativeModel) is False:
                raise TypeError("The type of `discriminative_model` must be `EBDiscriminativeModel`.")

        if generative_model is None:
            g_model = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                hidden_units_list=[
                    Conv2DTranspose(
                        channels=16,
                        kernel_size=6,
                        strides=(1, 1),
                        padding=(1, 1),
                    ), 
                    Conv2DTranspose(
                        channels=32,
                        kernel_size=6,
                        strides=(1, 1),
                        padding=(1, 1),
                    ), 
                    Conv2DTranspose(
                        channels=64,
                        kernel_size=6,
                        strides=(1, 1),
                        padding=(1, 1),
                    ), 
                    Conv2DTranspose(
                        channels=channel,
                        kernel_size=6,
                        strides=(1, 1),
                        padding=(1, 1),
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                hidden_dropout_rate_list=[0.5, 0.5, 0.5, 0.0],
                hidden_batch_norm_list=[BatchNorm(), BatchNorm(), BatchNorm(), None],
                optimizer_name="SGD",
                hidden_activation_list=["identity", "identity", "identity", "sigmoid"],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

            condition_sampler = ConditionSampler()
            condition_sampler.true_sampler = true_sampler

            generative_model = GenerativeModel(
                noise_sampler=UniformNoiseSampler(
                    low=-1e-05,
                    high=1e-05,
                    batch_size=batch_size,
                    seq_len=0,
                    channel=channel,
                    height=height,
                    width=width,
                    ctx=ctx
                ), 
                model=g_model, 
                initializer=None,
                condition_sampler=condition_sampler,
                conditonal_dim=1,
                learning_rate=learning_rate,
                optimizer_name="SGD",
                hybridize_flag=True,
                scale=1.0, 
                ctx=ctx, 
            )
        else:
            if isinstance(generative_model, GenerativeModel) is False:
                raise TypeError("The type of `generative_model` must be `GenerativeModel`.")

        EBGAN = EBGANController(
            true_sampler=true_sampler,
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            discriminator_loss=EBDiscriminatorLoss(weight=1.0),
            feature_matching_loss=L2NormLoss(weight=1.0),
            optimizer_name="SGD",
            learning_rate=learning_rate,
            learning_attenuate_rate=1.0,
            attenuate_epoch=50,
            hybridize_flag=True,
            scale=1.0,
            ctx=ctx,
            initializer=initializer,
        )
        self.EBGAN = EBGAN

    def learn(self, iter_n=1000, k_step=10):
        '''
        Learning.

        Args:
            iter_n:                         `int` of the number of training iterations.
            k_step:                         `int` of the number of learning of the `discriminative_model`.
        '''
        self.EBGAN.learn(
            iter_n=iter_n,
            k_step=k_step,
        )
