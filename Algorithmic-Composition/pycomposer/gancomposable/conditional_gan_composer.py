# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pycomposer.gan_composable import GANComposable

# MIDI controller.
from pycomposer.midi_controller import MidiController

# is-a `TrueSampler`
from pycomposer.truesampler.bar_gram_true_sampler import BarGramTrueSampler
# is-a `NoiseSampler`.
from pycomposer.noisesampler.bar_gram_noise_sampler import BarGramNoiseSampler
# n-gram of bars.
from pycomposer.bar_gram import BarGram

# is-a `NoiseSampler`.
from pygan.noisesampler.uniform_noise_sampler import UniformNoiseSampler
# is-a `GenerativeModel`.
from pygan.generativemodel.conditionalgenerativemodel.conditional_convolutional_model import ConditionalConvolutionalModel as Generator
# is-a `GenerativeModel`.
from pygan.generativemodel.deconvolution_model import DeconvolutionModel
# is-a `DiscriminativeModel`.
from pygan.discriminativemodel.cnnmodel.seq_cnn_model import SeqCNNModel as Discriminator
# is-a `GANsValueFunction`.
from pygan.gansvaluefunction.mini_max import MiniMax
# GANs framework.
from pygan.generative_adversarial_networks import GenerativeAdversarialNetworks
# Feature Matching.
from pygan.feature_matching import FeatureMatching

# Activation function.
from pydbm.activation.tanh_function import TanhFunction
# Activation function.
from pydbm.activation.softmax_function import SoftmaxFunction
# Batch normalization.
from pydbm.optimization.batch_norm import BatchNorm
# First convolution layer.
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer1
# Second convolution layer.
from pydbm.cnn.layerablecnn.convolution_layer import ConvolutionLayer as ConvolutionLayer2
# Computation graph in output layer.
from pydbm.synapse.cnn_output_graph import CNNOutputGraph
# Computation graph for first convolution layer.
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph1
# Computation graph for second convolution layer.
from pydbm.synapse.cnn_graph import CNNGraph as ConvGraph2
# Logistic Function as activation function.
from pydbm.activation.logistic_function import LogisticFunction
# Adams optimizer.
from pydbm.optimization.optparams.adam import Adam
# Convolutional Neural Networks(CNNs).
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork as CNN
# Cross entropy.
from pydbm.loss.cross_entropy import CrossEntropy
# Transposed convolution.
from pydbm.cnn.layerablecnn.convolutionlayer.deconvolution_layer import DeconvolutionLayer
# computation graph for transposed convolution.
from pydbm.synapse.cnn_graph import CNNGraph as DeCNNGraph


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
        hidden_dim=15200,
        generative_model=None,
        discriminative_model=None,
        gans_value_function=None
    ):
        '''
        Init.

        Args:
            midi_path_list:         `list` of paths to MIDI files.
            batch_size:             Batch size.
            seq_len:                The length of sequence that LSTM networks will observe.
            time_fraction:          Time fraction or time resolution (seconds).

            learning_rate:          Learning rate in `Generator` and `Discriminator`.

            hidden_dim:             The number of units in hidden layer of `DiscriminativeModel`.

            true_sampler:           is-a `TrueSampler`.
            noise_sampler:          is-a `NoiseSampler`.
            generative_model:       is-a `GenerativeModel`.
            discriminative_model:   is-a `DiscriminativeModel`.
            gans_value_function:    is-a `GANsValueFunction`.
        '''
        self.__midi_controller = MidiController()
        self.__midi_df_list = [self.__midi_controller.extract(midi_path) for midi_path in midi_path_list]

        bar_gram = BarGram(
            midi_df_list=self.__midi_df_list,
            time_fraction=time_fraction
        )
        self.__bar_gram = bar_gram
        dim = self.__bar_gram.dim

        true_sampler = BarGramTrueSampler(
            bar_gram=bar_gram,
            midi_df_list=self.__midi_df_list,
            batch_size=batch_size,
            seq_len=seq_len,
            time_fraction=time_fraction
        )

        noise_sampler = BarGramNoiseSampler(
            bar_gram=bar_gram,
            midi_df_list=self.__midi_df_list,
            batch_size=batch_size,
            seq_len=seq_len,
            time_fraction=time_fraction
        )

        if generative_model is None:
            conv_activation_function = TanhFunction()
            conv_activation_function.batch_norm = BatchNorm()

            channel = noise_sampler.channel

            convolution_layer_list = [
                ConvolutionLayer1(
                    ConvGraph1(
                        activation_function=conv_activation_function,
                        filter_num=batch_size,
                        channel=channel,
                        kernel_size=3,
                        scale=0.01,
                        stride=1,
                        pad=1
                    )
                )
            ]

            deconv_activation_function = SoftmaxFunction()

            deconvolution_layer_list = [
                DeconvolutionLayer(
                    DeCNNGraph(
                        activation_function=deconv_activation_function,
                        filter_num=batch_size,
                        channel=channel,
                        kernel_size=3,
                        scale=0.01,
                        stride=1,
                        pad=1
                    )
                )
            ]

            opt_params_deconv = Adam()
            deconvolution_model = DeconvolutionModel(
                deconvolution_layer_list=deconvolution_layer_list,
                opt_params=opt_params_deconv
            )

            opt_params=Adam()
            opt_params.dropout_rate = 0.0

            generative_model = Generator(
                batch_size=batch_size,
                layerable_cnn_list=convolution_layer_list,
                deconvolution_model=deconvolution_model,
                condition_noise_sampler=UniformNoiseSampler(
                    low=-0.1, 
                    high=0.1, 
                    output_shape=(batch_size, channel, seq_len, dim)
                ),
                learning_rate=learning_rate,
            )

        generative_model.noise_sampler = noise_sampler

        if discriminative_model is None:
            activation_function = LogisticFunction()
            activation_function.batch_norm = BatchNorm()

            # First convolution layer.
            conv2 = ConvolutionLayer2(
                # Computation graph for first convolution layer.
                ConvGraph2(
                    # Logistic function as activation function.
                    activation_function=activation_function,
                    # The number of `filter`.
                    filter_num=batch_size,
                    # The number of channel.
                    channel=noise_sampler.channel*2,
                    # The size of kernel.
                    kernel_size=3,
                    # The filter scale.
                    scale=0.001,
                    # The nubmer of stride.
                    stride=1,
                    # The number of zero-padding.
                    pad=1
                )
            )

            # Stack.
            layerable_cnn_list=[
                conv2
            ]

            opt_params = Adam()
            opt_params.dropout_rate = 0.0

            if hidden_dim is None:
                hidden_dim = channel * seq_len * dim

            cnn_output_activating_function = LogisticFunction()

            cnn_output_graph = CNNOutputGraph(
                hidden_dim=hidden_dim, 
                output_dim=1, 
                activating_function=cnn_output_activating_function, 
                scale=0.01
            )

            discriminative_model = Discriminator(
                batch_size=batch_size,
                layerable_cnn_list=layerable_cnn_list,
                cnn_output_graph=cnn_output_graph,
                opt_params=opt_params,
                computable_loss=CrossEntropy(),
                learning_rate=learning_rate
            )

        if gans_value_function is None:
            gans_value_function = MiniMax()

        GAN = GenerativeAdversarialNetworks(
            gans_value_function=gans_value_function,
            feature_matching=FeatureMatching(lambda1=0.01, lambda2=0.99)
        )

        self.__noise_sampler = noise_sampler
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
        generative_model, discriminative_model = self.__GAN.train(
            self.__true_sampler,
            self.__generative_model,
            self.__discriminative_model,
            iter_n=iter_n,
            k_step=k_step
        )
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model

    def extract_logs(self):
        '''
        Extract update logs data.

        Returns:
            The shape is:
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `discriminator`'s update turn.
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `generator`'s update turn.

        '''
        return self.__GAN.extract_logs_tuple()

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
        generated_arr = self.__generative_model.draw()
        channel = generated_arr.shape[1] // 2
        generated_arr = generated_arr[:, :channel]

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
                        program = self.__noise_sampler.program_list[program_key]
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
