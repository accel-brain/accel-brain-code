# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler
from pycomposer.midi_controller import MidiController


class MidiTrueSampler(TrueSampler):
    '''
    Sampler which draws samples from the `true` distribution of MIDI files.
    '''

    def __init__(
        self, 
        midi_path_list, 
        batch_size=20, 
        seq_len=10, 
        time_fraction=0.1
    ):
        '''
        Init.

        Args:
            midi_path_list:     `list` of paths to MIDI files.
            batch_size:         Batch size.
            seq_len:            The length of sequneces.
                                The length corresponds to the number of `time` splited by `time_fraction`.

            time_fraction:      Time fraction.
        '''
        midi_controller = MidiController()
        self.__midi_df_list = [None] * len(midi_path_list)
        for i in range(len(midi_path_list)):
            self.__midi_df_list[i] = midi_controller.extract(midi_path_list[i])

        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__time_fraction = time_fraction

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        sampled_arr = np.empty((self.__batch_size, self.__seq_len, 12))

        for batch in range(self.__batch_size):
            key = np.random.randint(low=0, high=len(self.__midi_df_list))
            midi_df = self.__midi_df_list[key]
            program_arr = midi_df.program.drop_duplicates().values
            key = np.random.randint(low=0, high=program_arr.shape[0])
            program_key = program_arr[key]
            midi_df = midi_df[midi_df.program == program_key]
            if midi_df.shape[0] < self.__seq_len:
                raise ValueError("The length of musical performance (program: " + str(program_key) + " is short.")

            row = np.random.uniform(
                low=midi_df.start.min(), 
                high=midi_df.end.max() - (self.__seq_len * self.__time_fraction)
            )
            for seq in range(self.__seq_len):
                df = midi_df[midi_df.start > row + (seq * self.__time_fraction)]
                df = df[df.end < row + ((seq+1) * self.__time_fraction)]
                sampled_arr[batch, seq] = self.__convert_into_feature(df)

        if sampled_arr.max() > sampled_arr.min():
            sampled_arr = (sampled_arr - sampled_arr.min()) / (sampled_arr.max() - sampled_arr.min())
        return sampled_arr

    def __convert_into_feature(self, df):
        arr = np.zeros(12)
        for i in range(df.shape[0]):
            arr[df.pitch.values[i] % 12] = 1

        return arr.reshape(1, -1).astype(float)
