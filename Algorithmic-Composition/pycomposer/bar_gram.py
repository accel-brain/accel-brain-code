# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class BarGram(object):
    '''
    The base class for n-gram representation of pitch in each bar.
    '''

    def __init__(
        self, 
        midi_df_list,
        time_fraction=0.1
    ):
        self.__midi_df_list = midi_df_list
        df = pd.concat(self.__midi_df_list)
        self.__min_pitch = df.pitch.min()
        self.__max_pitch = df.pitch.max()
        self.__min_velocity = df.velocity.min()
        self.__max_velocity = df.velocity.max()
        self.__program_list = df.program.drop_duplicates().values.tolist()
        self.__time_fraction = time_fraction

        self.__create_bar_gram()

    def extract_features(self, df):
        pitch_tuple = tuple(df.pitch.values.tolist())
        arr = np.zeros(self.__dim)
        try:
            arr[self.pitch_tuple_list.index(pitch_tuple)] = 1
        except ValueError:
            pitch_key = np.random.randint(low=0, high=len(pitch_tuple))
            arr[pitch_tuple[pitch_key]] = 1

        arr = arr.astype(float)
        return arr

    def __extract_bar_gram(self, midi_df):
        start = 0
        end = self.__time_fraction
        pitch_tuple_list = []
        while end < midi_df.end.max():
            df = midi_df[(start <= midi_df.start) & (midi_df.start <= end)]
            df = df[(df.start < end)]
            pitch_tuple = tuple(df.pitch.values.tolist())
            if pitch_tuple not in pitch_tuple_list:
                pitch_tuple_list.append(pitch_tuple)
            start += self.__time_fraction
            end += self.__time_fraction

        return pitch_tuple_list

    def __create_bar_gram(self):
        pitch_tuple_list = []
        for i in range(len(self.__midi_df_list)):
            pitch_tuple_list.extend(
                self.__extract_bar_gram(self.__midi_df_list[i])
            )
        df = pd.concat(self.__midi_df_list)
        pitch_list = [(v, ) for v in df.pitch.drop_duplicates().values.tolist()]
        pitch_tuple_list.extend(pitch_list)
        self.__dim = len(pitch_tuple_list)
        self.__pitch_tuple_list = pitch_tuple_list

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()

    def get_dim(self):
        ''' getter '''
        return self.__dim
    
    dim = property(get_dim, set_readonly)

    def get_pitch_tuple_list(self):
        ''' getter '''
        return self.__pitch_tuple_list
    
    pitch_tuple_list = property(get_pitch_tuple_list, set_readonly)
