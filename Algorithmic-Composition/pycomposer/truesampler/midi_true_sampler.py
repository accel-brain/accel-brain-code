# -*- coding: utf-8 -*-
import numpy as np
from pygan.true_sampler import TrueSampler


class MidiTrueSampler(TrueSampler):
    '''
    Sampler which draws samples from the `true` distribution of MIDI files.
    '''

    def __init__(
        self, 
        midi_df_list, 
        batch_size=20, 
        seq_len=10, 
        time_fraction=0.1,
        min_pitch=24,
        max_pitch=108
    ):
        '''
        Init.

        Args:
            midi_df_list:      `list` of paths to MIDI data extracted by `MidiController`.
            batch_size:         Batch size.
            seq_len:            The length of sequneces.
                                The length corresponds to the number of `time` splited by `time_fraction`.

            time_fraction:      Time fraction which means the length of bars.
            min_pitch:          The minimum of note number.
            max_pitch:          The maximum of note number.
        '''
        self.__midi_df_list = midi_df_list
        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__time_fraction = time_fraction
        self.__min_pitch = min_pitch
        self.__max_pitch = max_pitch
        self.__dim = self.__max_pitch - self.__min_pitch

    def draw(self):
        '''
        Draws samples from the `true` distribution.
        
        Returns:
            `np.ndarray` of samples.
        '''
        sampled_arr = np.empty((self.__batch_size, self.__seq_len, self.__dim))

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
                start = row + (seq * self.__time_fraction)
                end = row + ((seq+1) * self.__time_fraction)
                df = midi_df[(start <= midi_df.start) & (midi_df.start <= end)]
                sampled_arr[batch, seq] = self.__convert_into_feature(df)

        return sampled_arr

    def __convert_into_feature(self, df):
        arr = np.zeros(self.__dim)
        for i in range(df.shape[0]):
            if df.pitch.values[i] < self.__max_pitch - 1:
                if df.pitch.values[i] - self.__min_pitch >= 0:
                    arr[df.pitch.values[i] - self.__min_pitch] = 1

        return arr.reshape(1, -1).astype(float)
