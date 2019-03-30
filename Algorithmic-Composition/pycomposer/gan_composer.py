# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# MIDI controller.
from pycomposer.midi_controller import MidiController

# is-a `TrueSampler`
from pycomposer.truesampler.midi_true_sampler import MidiTrueSampler
# is-a `NoiseSampler`.
from pycomposer.noisesampler.midi_noise_sampler import MidiNoiseSampler
# is-a `GenerativeModel`.
from pygan.generativemodel.lstm_model import LSTMModel as Generator
# is-a `DiscriminativeModel`.
from pygan.discriminativemodel.lstm_model import LSTMModel as Discriminator
# is-a `GANsValueFunction`.
from pygan.gansvaluefunction.mini_max import MiniMax
# GANs framework.
from pygan.generative_adversarial_networks import GenerativeAdversarialNetworks
# The value function.
from pygan.gansvaluefunction.mini_max import MiniMax

# Activation function.
from pydbm.activation.tanh_function import TanhFunction
# Batch normalization.
from pydbm.optimization.batch_norm import BatchNorm


class GANComposer(object):
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

    References:
        - Fang, W., Zhang, F., Sheng, V. S., & Ding, Y. (2018). A method for improving CNN-based image recognition using DCGAN. Comput. Mater. Contin, 57, 167-178.
        - Gauthier, J. (2014). Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester, 2014(5), 2.
        - Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. (2014). Generative adversarial nets. In Advances in neural information processing systems (pp. 2672-2680).
        - Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).
        - Makhzani, A., Shlens, J., Jaitly, N., Goodfellow, I., & Frey, B. (2015). Adversarial autoencoders. arXiv preprint arXiv:1511.05644.

    '''

    def __init__(
        self, 
        midi_path_list, 
        target_program=0,
        batch_size=10,
        seq_len=4,
        time_fraction=1.0,
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
            target_program:         Program in generated MIDI.
            batch_size:             Batch size.
            seq_len:                The length of sequence that LSTM networks will observe.
            time_fraction:          Time fraction or time resolution (seconds).
            true_sampler:           is-a `TrueSampler`.
            noise_sampler:          is-a `NoiseSampler`.
            generative_model:       is-a `GenerativeModel`.
            discriminative_model:   is-a `DiscriminativeModel`.
            gans_value_function:    is-a `GANsValueFunction`.
        '''
        self.__midi_controller = MidiController()
        self.__midi_df_list = [self.__midi_controller.extract(midi_path) for midi_path in midi_path_list]

        # The dimension of observed or feature points.
        dim = 12

        if true_sampler is None:
            true_sampler = MidiTrueSampler(
                midi_path_list=midi_path_list,
                batch_size=batch_size,
                seq_len=seq_len
            )

        if noise_sampler is None:
            noise_sampler = MidiNoiseSampler(batch_size=batch_size)

        if generative_model is None:
            hidden_activating_function = TanhFunction()
            hidden_activating_function.batch_norm = BatchNorm()
            generative_model = Generator(
                batch_size=batch_size,
                seq_len=seq_len,
                input_neuron_count=dim,
                hidden_neuron_count=dim,
                hidden_activating_function=hidden_activating_function
            )

        generative_model.noise_sampler = noise_sampler

        if discriminative_model is None:
            discriminative_model = Discriminator(
                batch_size=batch_size,
                seq_len=seq_len,
                input_neuron_count=dim,
                hidden_neuron_count=dim
            )

        if gans_value_function is None:
            gans_value_function = MiniMax()

        GAN = GenerativeAdversarialNetworks(gans_value_function=gans_value_function)

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

    def compose(self, file_path, pitch_min=None, velocity_mean=None, velocity_std=None):
        '''
        Compose by learned model.

        Args:
            file_path:      Path to generated MIDI file.
            pitch_min:      Minimum of pitch.
                            This class generates the pitch in the range 
                            `pitch_min` to `pitch_min` + 12.
                            If `None`, the average pitch in MIDI files set to this parameter.

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
        if pitch_min is None:
            pitch_min = np.array(
                [self.__midi_df_list[i].pitch.mean() for i in range(len(self.__midi_df_list))]
            ).mean()
        if velocity_mean is None:
            velocity_mean = np.array(
                [self.__midi_df_list[i].velocity.mean() for i in range(len(self.__midi_df_list))]
            ).mean()
        if velocity_std is None:
            velocity_std = np.array(
                [self.__midi_df_list[i].velocity.std() for i in range(len(self.__midi_df_list))]
            ).mean()

        generated_list = []
        start = 0
        end = self.__time_fraction
        for batch in range(generated_arr.shape[0]):
            seq_arr, pitch_arr = np.where(generated_arr[batch] > generated_arr.mean())
            key_df = pd.DataFrame(
                np.c_[
                    seq_arr, 
                    pitch_arr, 
                    generated_arr[batch, seq_arr, pitch_arr]
                ], 
                columns=["seq", "pitch", "p"]
            )
            key_df = key_df.sort_values(by=["p"], ascending=False)
            for seq in range(generated_arr.shape[1]):
                df = key_df[key_df.seq == seq]
                for i in range(1):
                    pitch = int(df.pitch.values[i] + pitch_min)
                    velocity = np.random.normal(loc=velocity_mean, scale=velocity_std)
                    velocity = int(velocity)
                    generated_list.append((start, end, pitch, velocity))

                start += self.__time_fraction
                end += self.__time_fraction

        generated_midi_df = pd.DataFrame(generated_list, columns=["start", "end", "pitch", "velocity"])

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

        generated_midi_df["program"] = self.__target_program

        self.__midi_controller.save(
            file_path=file_path, 
            note_df=generated_midi_df
        )
