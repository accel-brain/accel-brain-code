# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import mxnet as mx
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm

from pycomposer.gan_composable import GANComposable

# MIDI controller.
from pycomposer.midi_controller import MidiController

# is-a `TrueSampler`
from pycomposer.samplabledata.truesampler._mxnet.bar_gram_true_sampler import BarGramTrueSampler
from pycomposer.samplabledata.truesampler._mxnet.bargramtruesampler.conditonal_bar_gram_true_sampler import ConditionalBarGramTrueSampler

# n-gram of bars.
from pycomposer.bar_gram import BarGram

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata._mxnet.image_extractor import ImageExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_image_iterator import UnlabeledImageIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutional_auto_encoder import ConvolutionalAutoEncoder
from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController



class ConditionalGANComposer(GANComposable):
    '''
    Algorithmic Composer based on Conditional Generative Adversarial Networks(Conditional GANs).

    This composer learns observed data points drawn from a conditional true distribution 
    of input MIDI files and generates feature points drawn from a fake distribution 
    that means such as Uniform distribution or Normal distribution, imitating the true MIDI 
    files data.

    The components included in this class are functionally differentiated into three models.

    1. `TrueSampler`.
    2. `Generator`.
    3. `Discriminator`.

    The function of `TrueSampler` is to draw samples from a true distribution of input MIDI files. 
    `Generator` has `NoiseSampler`s which can be considered as a `Conditioner`s like the 
    MidiNet(Yang, L. C., et al., 2017) and draw fake samples from a Uniform distribution or Normal 
    distribution by use it. And `Discriminator` observes those input samples, trying discriminating 
    true and fake data. 

    While `Discriminator` observes `Generator`'s observation to discrimine the output from true samples, 
    `Generator` observes `Discriminator`'s observations to confuse `Discriminator`s judgments. 
    In GANs framework, the mini-max game can be configured by the observations of observations.

    After this game, the `Generator` will grow into a functional equivalent that enables to imitate 
    the `TrueSampler` and makes it possible to compose similar but slightly different music by the 
    imitation.

    In this class, Convolutional Neural Networks(CNNs) and Deconvolution Networks are implemented as 
    `Generator` and `Discriminator`. The Deconvolution also called transposed convolutions "work by 
    swapping the forward and backward passes of a convolution." (Dumoulin, V., & Visin, F. 2016, p20.)

    Following MidiNet and MuseGAN(Dong, H. W., et al., 2018), this class consider bars
    as the basic compositional unit for the fact that harmonic changes usually occur at 
    the boundaries of bars and that human beings often use bars as the building blocks 
    when composing songs. The feature engineering in this class also is inspired by 
    the Multi-track piano-roll representations in MuseGAN. But their strategies of 
    activation function did not apply to this library since its methods can cause 
    information losses. The models just binarize the `Generator`'s output, which 
    uses tanh as an activation function in the output layer, by a threshold at zero, 
    or by deterministic or stochastic binary neurons(Bengio, Y., et al., 2018, Chung, J., et al., 2016), 
    and ignore drawing a distinction the consonance and the dissonance.

    This library simply uses the softmax strategy. This class stochastically selects 
    a combination of pitches in each bars drawn by the true MIDI files data, based on 
    the difference between consonance and dissonance intended by the composer of the MIDI files.

    References:
        - Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432.
        - Chung, J., Ahn, S., & Bengio, Y. (2016). Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704.
        - Dong, H. W., Hsiao, W. Y., Yang, L. C., & Yang, Y. H. (2018, April). MuseGAN: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. In Thirty-Second AAAI Conference on Artificial Intelligence.
        - Dumoulin, V., & V,kisin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
        - Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
        - Yang, L. C., Chou, S. Y., & Yang, Y. H. (2017). MidiNet: A convolutional generative adversarial network for symbolic-domain music generation. arXiv preprint arXiv:1703.10847.
    '''

    def __init__(
        self, 
        midi_path_list, 
        batch_size=20,
        seq_len=8,
        time_fraction=1.0,
        learning_rate=1e-10,
        learning_attenuate_rate=0.1,
        attenuate_epoch=50,
        generative_model=None,
        discriminative_model=None,
        ctx=mx.gpu(),
        initializer=None,
    ):
        '''
        Init.

        Args:
            midi_path_list:                 `list` of paths to MIDI files.
            batch_size:                     Batch size.
            seq_len:                        The length of sequence that LSTM networks will observe.
            time_fraction:                  Time fraction or time resolution (seconds).

            learning_rate:                  Learning rate in `Generator` and `Discriminator`.
            learning_attenuate_rate:        Attenuate the `learning_rate` by a factor of this value every `attenuate_epoch`.
            attenuate_epoch:                Attenuate the `learning_rate` by a factor of `learning_attenuate_rate` every `attenuate_epoch`.

            true_sampler:                   is-a `TrueSampler`.
            noise_sampler:                  is-a `NoiseSampler`.
            generative_model:               is-a `GenerativeModel`.
            discriminative_model:           is-a `DiscriminativeModel`.
            ctx:                            `mx.cpu()` or `mx.gpu()`.
            initializer:                    is-a `mxnet.initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
        '''
        computable_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)

        self.__midi_controller = MidiController()
        self.__midi_df_list = [self.__midi_controller.extract(midi_path) for midi_path in midi_path_list]

        bar_gram = BarGram(
            midi_df_list=self.__midi_df_list,
            time_fraction=time_fraction
        )
        self.__bar_gram = bar_gram
        dim = self.__bar_gram.dim

        c_true_sampler = ConditionalBarGramTrueSampler(
            bar_gram=bar_gram,
            midi_df_list=self.__midi_df_list,
            batch_size=batch_size,
            seq_len=seq_len,
            time_fraction=time_fraction
        )

        true_sampler = BarGramTrueSampler(
            bar_gram=bar_gram,
            midi_df_list=self.__midi_df_list,
            batch_size=batch_size,
            seq_len=seq_len,
            time_fraction=time_fraction
        )

        if generative_model is None:
            condition_sampler = ConditionSampler()
            condition_sampler.true_sampler = true_sampler

            c_model = ConvolutionalNeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                hidden_units_list=[
                    Conv2D(
                        channels=16,
                        kernel_size=6,
                        strides=(1, 1),
                        padding=(1, 1),
                    ), 
                    Conv2D(
                        channels=len(true_sampler.program_list),
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
                hidden_dropout_rate_list=[0.5, 0.0],
                hidden_batch_norm_list=[BatchNorm(), None],
                optimizer_name="SGD",
                hidden_activation_list=["relu", "identity"],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )
            condition_sampler.model = c_model
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
                        channels=len(true_sampler.program_list),
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
                hidden_dropout_rate_list=[0.5, 0.0],
                hidden_batch_norm_list=[BatchNorm(), None],
                optimizer_name="SGD",
                hidden_activation_list=["relu", "identity"],
                hidden_residual_flag=False,
                hidden_dense_flag=False,
                dense_axis=1,
                ctx=ctx,
                hybridize_flag=True,
                regularizatable_data_list=[],
                scale=1.0,
            )

            generative_model = GenerativeModel(
                noise_sampler=None, 
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

        if discriminative_model is None:
            output_nn = NeuralNetworks(
                computable_loss=computable_loss,
                initializer=initializer,
                learning_rate=learning_rate,
                learning_attenuate_rate=1.0,
                attenuate_epoch=50,
                units_list=[100, 1],
                dropout_rate_list=[0.5, 0.0],
                optimizer_name="SGD",
                activation_list=["relu", "sigmoid"],
                hidden_batch_norm_list=[BatchNorm(), None],
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

        GAN = GANController(
            true_sampler=c_true_sampler,
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            generator_loss=GeneratorLoss(weight=1.0),
            discriminator_loss=DiscriminatorLoss(weight=1.0),
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

        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__GAN = GAN
        self.__time_fraction = time_fraction

    def learn(self, iter_n=500, k_step=10):
        '''
        Learning.

        Args:
            iter_n:     The number of training iterations.
            k_step:     The number of learning of the `discriminator`.

        '''
        self.__GAN.learn(
            iter_n=iter_n,
            k_step=k_step
        )
        self.__generative_model = self.__GAN.generative_model
        self.__discriminative_model = self.__GAN.discriminative_model

    def extract_logs(self):
        '''
        Extract update logs data.

        Returns:
            The shape is:
            - `np.ndarray` of probabilities inferenced by the `discriminator` (mean) in the `discriminator`'s update turn.
            - `np.ndarray` of probabilities inferenced by the `discriminator` (mean) in the `generator`'s update turn.
            - `np.ndarray` of posterior(mean).
            - `np.ndarray` of losses of the feature matching.

        '''
        return (
            self.__GAN.generative_loss_arr,
            self.__GAN.discriminative_loss_arr,
            self.__GAN.posterior_logs_arr,
            self.__GAN.feature_matching_loss_arr
        )

    def compose(self, file_path, velocity_mean=None, velocity_std=None):
        '''
        Compose by learned model.

        Args:
            file_path:      Path to generated MIDI file.

            velocity_mean:  Mean of velocity.
                            This class samples the velocity from a Gaussian distribution of 
                            `velocity_mean` and `velocity_std`.
                            If `None`, the average velocity in MIDI files set to this parameter.

            velocity_std:   Standard deviation(SD) of velocity.
                            This class samples the velocity from a Gaussian distribution of 
                            `velocity_mean` and `velocity_std`.
                            If `None`, the SD of velocity in MIDI files set to this parameter.
        '''
        generated_arr = self.__generative_model.draw().asnumpy()
        channel = generated_arr.shape[1] // 2
        generated_arr = generated_arr[:, channel:]

        # @TODO(chimera0(RUM)): Fix the redundant processings.
        if velocity_mean is None:
            velocity_mean = np.array(
                [self.__midi_df_list[i].velocity.mean() for i in range(len(self.__midi_df_list))]
            ).mean()
        if velocity_std is None:
            velocity_std = np.array(
                [self.__midi_df_list[i].velocity.std() for i in range(len(self.__midi_df_list))]
            ).std()

        generated_list = []
        start = 0
        end = self.__time_fraction
        for batch in range(generated_arr.shape[0]):
            for seq in range(generated_arr.shape[2]):
                add_flag = False
                for program_key in range(generated_arr.shape[1]):
                    pitch_key = np.argmax(generated_arr[batch, program_key, seq])
                    pitch_tuple = self.__bar_gram.pitch_tuple_list[pitch_key]
                    for pitch in pitch_tuple:
                        velocity = np.random.normal(
                            loc=velocity_mean, 
                            scale=velocity_std
                        )
                        velocity = int(velocity)
                        program = self.__true_sampler.program_list[program_key]
                        generated_list.append((program, start, end, pitch, velocity))
                        add_flag = True

                if add_flag is True:
                    start += self.__time_fraction
                    end += self.__time_fraction

        generated_midi_df = pd.DataFrame(
            generated_list, 
            columns=[
                "program",
                "start", 
                "end", 
                "pitch", 
                "velocity"
            ]
        )

        pitch_arr = generated_midi_df.pitch.drop_duplicates()
        df_list = []
        for pitch in pitch_arr:
            df = generated_midi_df[generated_midi_df.pitch == pitch]
            df = df.sort_values(by=["start", "end"])
            df["next_start"] = df.start.shift(-1)
            df["next_end"] = df.end.shift(-1)
            df.loc[df.end == df.next_start, "end"] = df.loc[df.end == df.next_start, "next_end"]
            df = df.drop_duplicates(["end"])
            df_list.append(df)

        generated_midi_df = pd.concat(df_list)
        generated_midi_df = generated_midi_df.sort_values(by=["start", "end"])

        self.__midi_controller.save(
            file_path=file_path, 
            note_df=generated_midi_df
        )

    def get_generative_model(self):
        ''' getter '''
        return self.__generative_model
    
    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")
    
    generative_model = property(get_generative_model, set_readonly)

    def get_true_sampler(self):
        ''' getter '''
        return self.__true_sampler
    
    true_sampler = property(get_true_sampler, set_readonly)

    def get_bar_gram(self):
        ''' getter '''
        return self.__bar_gram

    bar_gram = property(get_bar_gram, set_readonly)
