# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# MIDI controller.
from pycomposer.midi_controller import MidiController

# is-a `TrueSampler`
from pycomposer.truesampler.bar_true_sampler import BarTrueSampler
# is-a `NoiseSampler`.
from pycomposer.noisesampler.bar_noise_sampler import BarNoiseSampler

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

# Activation function.
from pydbm.activation.tanh_function import TanhFunction
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
# Tanh Function as activation function.
from pydbm.activation.tanh_function import TanhFunction
# ReLu Function as activation function.
from pydbm.activation.relu_function import ReLuFunction
# Identity function as activation function.
from pydbm.activation.identity_function import IdentityFunction
# Sign function as activation function.
from pydbm.activation.signfunction.deterministic_binary_neurons import DeterministicBinaryNeurons
from pydbm.activation.signfunction.stochastic_binary_neurons import StochasticBinaryNeurons
# SGD optimizer.
from pydbm.optimization.optparams.sgd import SGD
# Adams optimizer.
from pydbm.optimization.optparams.adam import Adam
# Convolutional Neural Networks(CNNs).
from pydbm.cnn.convolutional_neural_network import ConvolutionalNeuralNetwork as CNN
# Mean Squared Error(MSE).
from pydbm.loss.mean_squared_error import MeanSquaredError
from pydbm.loss.cross_entropy import CrossEntropy
# Transposed convolution.
from pydbm.cnn.layerablecnn.convolutionlayer.deconvolution_layer import DeconvolutionLayer
# computation graph for transposed convolution.
from pydbm.synapse.cnn_graph import CNNGraph as DeCNNGraph
# Verification.
from pydbm.verification.verificate_function_approximation import VerificateFunctionApproximation


class ConditionalGANComposer(object):
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

    Following MidiNet and MuseGAN(Dong, H. W., et al., 2018), this class consider bars
    as the basic compositional unit for the fact that harmonic changes usually occur at 
    the boundaries of bars and that human beings often use bars as the building blocks 
    when composing songs. The feature engineering in this class also is inspired by 
    the Multi-track piano-roll representations in MuseGAN. 

    References:
        - Dong, H. W., Hsiao, W. Y., Yang, L. C., & Yang, Y. H. (2018, April). MuseGAN: Multi-track sequential generative adversarial networks for symbolic music generation and accompaniment. In Thirty-Second AAAI Conference on Artificial Intelligence.
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
        batch_size=10,
        seq_len=4,
        time_fraction=1.0,
        min_pitch=24,
        max_pitch=108,
        learning_rate=1e-05,
        hidden_dim=None,
        true_sampler=None,
        noise_sampler=None,
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

            min_pitch:              The minimum of note number.
            max_pitch:              The maximum of note number.

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

        # The dimension of observed or feature points.
        dim = max_pitch - min_pitch

        if true_sampler is None:
            true_sampler = BarTrueSampler(
                midi_df_list=self.__midi_df_list,
                batch_size=batch_size,
                seq_len=seq_len,
                time_fraction=time_fraction,
                min_pitch=min_pitch,
                max_pitch=max_pitch
            )

        if noise_sampler is None:
            noise_sampler = BarNoiseSampler(
                midi_df_list=self.__midi_df_list,
                batch_size=batch_size,
                seq_len=seq_len,
                time_fraction=time_fraction,
                min_pitch=min_pitch,
                max_pitch=max_pitch
            )

        if generative_model is None:
            conv_activation_function = LogisticFunction()
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

            deconv_activation_function = DeterministicBinaryNeurons()

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
                opt_params=opt_params_deconv,
                verbose_mode=False
            )

            opt_params=Adam()
            opt_params.dropout_rate = 0.0

            generative_model = Generator(
                batch_size=batch_size,
                layerable_cnn_list=convolution_layer_list,
                deconvolution_model=deconvolution_model,
                conditon_noise_sampler=UniformNoiseSampler(
                    low=0, 
                    high=1, 
                    output_shape=(batch_size, channel, seq_len, dim)
                ),
                learning_rate=learning_rate,
                verbose_mode=False
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
                hidden_dim = (noise_sampler.channel * 2) * seq_len * dim

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
                learning_rate=learning_rate,
                verbose_mode=False
            )

        if gans_value_function is None:
            gans_value_function = MiniMax()

        GAN = GenerativeAdversarialNetworks(gans_value_function=gans_value_function)

        self.__noise_sampler = noise_sampler
        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__GAN = GAN
        self.__time_fraction = time_fraction
        self.__min_pitch = min_pitch
        self.__max_pitch = max_pitch

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
            for program_key in range(generated_arr.shape[1]):
                seq_arr, pitch_arr = np.where(generated_arr[batch, program_key] == 1)
                key_df = pd.DataFrame(
                    np.c_[
                        seq_arr, 
                        pitch_arr, 
                        generated_arr[batch, program_key, seq_arr, pitch_arr]
                    ], 
                    columns=["seq", "pitch", "p"]
                )
                key_df = key_df.sort_values(by=["p"], ascending=False)
                program = self.__noise_sampler.program_list[program_key]
                for seq in range(generated_arr.shape[2]):
                    df = key_df[key_df.seq == seq]
                    for i in range(df.shape[0]):
                        pitch = int(df.pitch.values[i] + self.__min_pitch)
                        velocity = np.random.normal(loc=velocity_mean, scale=velocity_std)
                        velocity = int(velocity)
                        generated_list.append((program, start, end, pitch, velocity))

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
