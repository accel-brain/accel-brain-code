# -*- coding: utf-8 -*-
from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata._mxnet.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_image_iterator import UnlabeledImageIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks

from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutionalautoencoder.contractive_cae import ContractiveCAE as ConvolutionalAutoEncoder
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generativemodel.convolutional_auto_encoder import ConvolutionalAutoEncoder as GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler

from pygan.controllablemodel._mxnet.gancontroller.ganomaly_controller import GanomalyController


import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.loss import L1Loss


class GANomalyDetector(object):
    '''
    GANomaly, Semi-Supervised Anomaly Detection via Adversarial Training.

    This is the beta version.

    GANomaly is a model of semi-supervised anomaly detection, which is a 
    novel adversarial autoencoder within an encoder-decoder-encoder pipeline, 
    capturing the training data distribution within both image and latent vector 
    space, yielding superior results to contemporary GAN-based and traditional 
    autoencoder-based approaches.

    References:
        - Akcay, S., Atapour-Abarghouei, A., & Breckon, T. P. (2018, December). Ganomaly: Semi-supervised anomaly detection via adversarial training. In Asian Conference on Computer Vision (pp. 622-637). Springer, Cham.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
        - Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training gans. In Advances in neural information processing systems (pp. 2234-2242).

    '''

    def __init__(
        self,
        dir_list,
        test_dir_list,
        width=28,
        height=28,
        channel=1,
        initializer=None,
        batch_size=40,
        learning_rate=0.0002,
        ctx=mx.gpu(),
        discriminative_model=None,
        generative_model=None,
        re_encoder_model=None,
        advarsarial_loss_weight=1.0,
        encoding_loss_weight=1.0,
        contextual_loss_weight=1.0,
        discriminator_loss_weight=1.0,
    ):
        '''
        Init.

        If you are not satisfied with this simple default setting,
        delegate `discriminative_model` and `generative_model` designed by yourself.

        Args:
            dir_list:       `list` of `str` of path to image files.
            test_dir_list:  `list` of `str` of path to image files for test.
            width:          `int` of image width.
            height:         `int` of image height.
            channel:        `int` of image channel.
            initializer:    is-a `mxnet.initializer` for parameters of model.
                            If `None`, it is drawing from the Xavier distribution.
            
            batch_size:     `int` of batch size.
            learning_rate:  `float` of learning rate.
            ctx:            `mx.gpu()` or `mx.cpu()`.

            discriminative_model:       is-a `accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model.DiscriminativeModel`.
            generative_model:           is-a `accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model.GenerativeModel`.
            re_encoder_model:           is-a `HybridBlock`.

            advarsarial_loss_weight:    `float` of weight for advarsarial loss.
            encoding_loss_weight:       `float` of weight for encoding loss.
            contextual_loss_weight:     `float` of weight for contextual loss.
            discriminator_loss_weight:  `float` of weight for discriminator loss.
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
            scale=1/1000,
            noiseable_data=GaussNoise(sigma=1e-08, mu=0.0),
        )

        test_unlabeled_image_iterator = UnlabeledImageIterator(
            image_extractor=image_extractor,
            dir_list=test_dir_list,
            batch_size=batch_size,
            norm_mode="z_score",
            scale=1/1000,
            noiseable_data=GaussNoise(sigma=1e-08, mu=0.0),
        )

        true_sampler = TrueSampler()
        true_sampler.iteratorable_data = unlabeled_image_iterator

        condition_sampler = ConditionSampler()
        condition_sampler.true_sampler = true_sampler

        computable_loss = L2NormLoss()

        if initializer is None:
            initializer = mx.initializer.Uniform()
        else:
            if isinstance(initializer, mx.initializer.Initializer) is False:
                raise TypeError("The type of `initializer` must be `mxnet.initializer.Initializer`.")

        if discriminative_model is None:
            output_nn = NeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                units_list=[1],
                dropout_rate_list=[0.0],
                optimizer_name="SGD",
                activation_list=["sigmoid"],
                hidden_batch_norm_list=[None],
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
                output_no_bias_flag=True,
                all_no_bias_flag=True,
                not_init_flag=False,
            )

            d_model = ConvolutionalNeuralNetworks(
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
                        strides=(2, 2),
                        padding=(1, 1),
                    ),
                ],
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=output_nn,
                hidden_dropout_rate_list=[0.5, 0.5,],
                hidden_batch_norm_list=[BatchNorm(), BatchNorm()],
                optimizer_name="SGD",
                hidden_activation_list=["relu", "relu",],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

            discriminative_model = DiscriminativeModel(
                model=d_model, 
                initializer=None,
                learning_rate=learning_rate,
                optimizer_name="SGD",
                hybridize_flag=True,
                scale=1.0, 
                ctx=ctx, 
            )
        else:
            if isinstance(discriminative_model, DiscriminativeModel) is False:
                raise TypeError("The type of `discriminative_model` must be `DiscriminativeModel`.")

        if re_encoder_model is None:
            re_encoder_model = ConvolutionalNeuralNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `list` of int` of the number of units in hidden layers.
                hidden_units_list=[
                    # `mxnet.gluon.nn.Conv2D`.
                    Conv2D(
                        channels=16,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ), 
                    Conv2D(
                        channels=32,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ),
                ],
                # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                hidden_activation_list=["relu", "relu",],
                # `list` of `float` of dropout rate.
                hidden_dropout_rate_list=[0.5, 0.5, ],
                # `list` of `mxnet.gluon.nn.BatchNorm`.
                hidden_batch_norm_list=[BatchNorm(), BatchNorm()],
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )

        if generative_model is None:
            encoder = ConvolutionalNeuralNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `list` of int` of the number of units in hidden layers.
                hidden_units_list=[
                    # `mxnet.gluon.nn.Conv2D`.
                    Conv2D(
                        channels=16,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ), 
                    Conv2D(
                        channels=32,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ),
                ],
                # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                hidden_activation_list=["relu", "relu", ],
                # `list` of `float` of dropout rate.
                hidden_dropout_rate_list=[0.5, 0.5, ],
                # `list` of `mxnet.gluon.nn.BatchNorm`.
                hidden_batch_norm_list=[BatchNorm(), BatchNorm()],
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )
            decoder = ConvolutionalNeuralNetworks(
                # is-a `ComputableLoss` or `mxnet.gluon.loss`.
                computable_loss=computable_loss,
                # `list` of int` of the number of units in hidden layers.
                hidden_units_list=[
                    Conv2DTranspose(
                        channels=16,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(1, 1),
                    ), 
                    Conv2DTranspose(
                        channels=channel,
                        kernel_size=6,
                        strides=(2, 2),
                        padding=(0, 0),
                    ),
                ],
                # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
                hidden_activation_list=["relu", "tanh"],
                # `list` of `float` of dropout rate.
                hidden_dropout_rate_list=[0.5, 0.0],
                # `list` of `mxnet.gluon.nn.BatchNorm`.
                hidden_batch_norm_list=[BatchNorm(), None],
                # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
                hybridize_flag=True,
                # `mx.gpu()` or `mx.cpu()`.
                ctx=ctx,
            )

            g_model = ConvolutionalAutoEncoder(
                # is-a `ConvolutionalNeuralNetworks`.
                encoder=encoder,
                # is-a `ConvolutionalNeuralNetworks`.
                decoder=decoder,
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                input_nn=None,
                input_result_height=None,
                input_result_width=None,
                input_result_channel=None,
                output_nn=None,
                optimizer_name="SGD",
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

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

        ganomaly_controller = GanomalyController(
            generative_model=generative_model,
            re_encoder_model=re_encoder_model,
            discriminative_model=discriminative_model,
            advarsarial_loss=L2NormLoss(weight=advarsarial_loss_weight),
            encoding_loss=L2NormLoss(weight=encoding_loss_weight),
            contextual_loss=L1Loss(weight=contextual_loss_weight),
            discriminator_loss=DiscriminatorLoss(weight=discriminator_loss_weight),
            feature_matching_loss=None,
            optimizer_name="SGD",
            learning_rate=learning_rate,
            learning_attenuate_rate=1.0,
            attenuate_epoch=50,
            hybridize_flag=True,
            scale=1.0,
            ctx=ctx,
            initializer=initializer,
        )

        self.ganomaly_controller = ganomaly_controller
        self.test_unlabeled_image_iterator = test_unlabeled_image_iterator

    def learn(self, iter_n=1000, k_step=10):
        '''
        Learning.

        Args:
            iter_n:                         `int` of the number of training iterations.
            k_step:                         `int` of the number of learning of the `discriminative_model`.
        '''
        self.ganomaly_controller.learn(
            iter_n=iter_n,
            k_step=k_step,
        )

    def infernce(self):
        file_path_list = []
        anomaly_score_arr = None
        for _, _, test_batch_arr, _file_path_list in self.test_unlabeled_image_iterator.generate_inferenced_samples():
            _anomaly_score_arr = self.ganomaly_controller.infernce_anomaly_score(test_batch_arr)
            
            if anomaly_score_arr is None:
                anomaly_score_arr = _anomaly_score_arr
            else:
                anomaly_score_arr = nd.concat(anomaly_score_arr, _anomaly_score_arr, dim=0)
            file_path_list.extend(_file_path_list)
        
        anomaly_score_arr = (anomaly_score_arr - anomaly_score_arr.min()) / (anomaly_score_arr.max() - anomaly_score_arr.min())
        return np.c_[anomaly_score_arr.asnumpy(), np.array(file_path_list)]
