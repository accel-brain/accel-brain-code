# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pycomposer.gan_composable import GANComposable

# MIDI controller.
from pycomposer.midi_controller import MidiController
# n-gram of bars.
from pycomposer.bar_gram import BarGram

# is-a `GenerativeModel`.
from pygan.generativemodel.lstm_model import LSTMModel as Generator
# is-a `DiscriminativeModel`.
from pygan.discriminativemodel.lstm_model import LSTMModel as Discriminator
# is-a `GANsValueFunction`.
from pygan.gansvaluefunction.mini_max import MiniMax
# GANs framework.
from pygan.generative_adversarial_networks import GenerativeAdversarialNetworks
# Feature Matching.
from pygan.feature_matching import FeatureMatching

# Activation function.
from pydbm.activation.tanh_function import TanhFunction
# Batch normalization.
from pydbm.optimization.batch_norm import BatchNorm


class GANComposer(GANComposable):
    '''
    Algorithmic Composer based on Generative Adversarial Networks(GANs).

    This composer learns observed data points drawn from a true distribution 
    of input MIDI files and generates feature points drawn from a fake distribution 
    that means such as Uniform distribution or Normal distribution, imitating the true MIDI 
    files data.

    The components included in this class are functionally differentiated into three models.

    1. `TrueSampler`.
    2. `Generator`.
    3. `Discriminator`.

    The function of `TrueSampler` is to draw samples from a true distribution of input MIDI files. 
    `Generator` has `NoiseSampler`s and draw fake samples from a Uniform distribution or Normal 
    distribution by use it. And `Discriminator` observes those input samples, trying discriminating 
    true and fake data. 

    While `Discriminator` observes `Generator`'s observation to discrimine the output from true samples, 
    `Generator` observes `Discriminator`'s observations to confuse `Discriminator`s judgments. 
    In GANs framework, the mini-max game can be configured by the observations of observations.

    After this game, the `Generator` will grow into a functional equivalent that enables to imitate 
    the `TrueSampler` and makes it possible to compose similar but slightly different music by the 
    imitation.

    In this class, Long short term memory(LSTM) networks are implemented as `Generator` and `Discriminator`.
    Originally, Long Short-Term Memory(LSTM) networks as a special RNN structure has proven stable and 
    powerful for modeling long-range dependencies.
    
    The Key point of structural expansion is its memory cell which essentially acts as an accumulator of the state information. 
    Every time observed data points are given as new information and input to LSTM's input gate, its information will be accumulated to 
    the cell if the input gate is activated. The past state of cell could be forgotten in this process if LSTM's forget gate is on.
    Whether the latest cell output will be propagated to the final state is further controlled by the output gate.

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
        - Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.
        - Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). LSTM-based encoder-decoder for multi-sensor anomaly detection. arXiv preprint arXiv:1607.00148.
        - Zaremba, W., Sutskever, I., & Vinyals, O. (2014). Recurrent neural network regularization. arXiv preprint arXiv:1409.2329.

    '''

    def __init__(
        self, 
        midi_path_list, 
        target_program=0,
        batch_size=20,
        seq_len=8,
        time_fraction=1.0,
        learning_rate=1e-10,
        generative_model=None,
        discriminative_model=None,
        gans_value_function=None
    ):
        '''
        Init.

        Args:
            midi_path_list:         `list` of paths to MIDI files.
            target_program:         Program in generated MIDI.
            batch_size:             Batch size.
            seq_len:                The length of sequence that LSTM networks will observe.
            time_fraction:          Time fraction or time resolution (seconds).
            learning_rate:          Learning rate in `Generator` and `Discriminator`.
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
            hidden_activating_function = TanhFunction()
            hidden_activating_function.batch_norm = BatchNorm()
            output_gate_activating_function = SoftmaxFunction()
            generative_model = Generator(
                batch_size=batch_size,
                seq_len=seq_len,
                input_neuron_count=dim,
                hidden_neuron_count=dim,
                output_gate_activating_function=output_gate_activating_function,
                hidden_activating_function=hidden_activating_function,
                learning_rate=learning_rate
            )

        generative_model.noise_sampler = noise_sampler

        if discriminative_model is None:
            discriminative_model = Discriminator(
                batch_size=batch_size,
                seq_len=seq_len,
                input_neuron_count=dim,
                hidden_neuron_count=dim,
                learning_rate=learning_rate
            )

        if gans_value_function is None:
            gans_value_function = MiniMax()

        GAN = GenerativeAdversarialNetworks(
            gans_value_function=gans_value_function,
            feature_matching=FeatureMatching(lambda1=0.01, lambda2=0.99)
        )

        self.__true_sampler = true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__GAN = GAN
        self.__time_fraction = time_fraction
        self.__target_program = target_program

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
                pitch_key = np.argmax(generated_arr[batch, seq])
                pitch_tuple = self.__bar_gram.pitch_tuple_list[pitch_key]
                for pitch in pitch_tuple:
                    velocity = np.random.normal(
                        loc=velocity_mean, 
                        scale=velocity_std
                    )
                    velocity = int(velocity)
                    generated_list.append((start, end, pitch, velocity))
                    add_flag = True

                if add_flag is True:
                    start += self.__time_fraction
                    end += self.__time_fraction

        generated_midi_df = pd.DataFrame(generated_list, columns=["start", "end", "pitch", "velocity"])
        generated_midi_df["program"] = self.__target_program

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
