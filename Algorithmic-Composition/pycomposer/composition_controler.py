# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pretty_midi
from pycomposer.chord_progression import ChordProgression
from pycomposer.melody_composer import MelodyComposer


class CompositionControler(object):

    # The object of `pretty_midi.PrettyMIDI`.
    __pm = None
    # The object of `pretty_midi.Instrument`.
    __instrument_chord = None
    # The object of `pretty_midi.Instrument`.
    __instrument_melody = None
    # The object of `ChordProgression`.
    __chord_progression = None
    # The object of `MelodyComposer`.
    __melody_composer = None
    # Now time.
    __now_time = 0.0
    
    __pitch_max = None
    __pitch_min = None
    __pre_pitch_n = 5

    def __init__(
        self,
        rbm,
        resolution=960,
        initial_tempo=120,
        chord_instrument_num=39,
        melody_instrument_num=0
    ):
        self.reset(
            resolution,
            initial_tempo,
            chord_instrument_num,
            melody_instrument_num        
        )
        self.__chord_progression = ChordProgression()
        self.__melody_composer = MelodyComposer()
        self.__rbm = rbm

    def reset(
        self,
        resolution=960,
        initial_tempo=120,
        chord_instrument_num=39,
        melody_instrument_num=0
    ):
        self.__pm = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
        self.__instrument_chord = pretty_midi.Instrument(chord_instrument_num)
        self.__instrument_melody = pretty_midi.Instrument(melody_instrument_num)

    def learn_pitch(self, tone_df, training_count, batch_size):
        for i in range(tone_df.shape[0]):
            pitch_arr = np.zeros(127)
            pitch_arr[tone_df.pitch.values[i]] = 1
            pitch_arr = pitch_arr.astype(np.float64)
            self.__rbm.approximate_learning(
                pitch_arr,
                traning_count=training_count, 
                batch_size=batch_size
            )
        self.__training_count = training_count
        self.__r_batch_size = batch_size

    def create_chord_melody_list(
        self,
        octave,
        first_chord="I",
        length=8
    ):
        chord_melody_list = [self.__melody_composer.create(first_chord)]
        for i in range(length):
            chord = self.__chord_progression.progression(state=first_chord, octave=octave)
            chord_melody_list.append(self.__melody_composer.create(chord))

        return chord_melody_list

    def match_melody_to_chords(
        self,
        chord_melody_list,
        chord_duration_range=(0.75, 1),
        start_time=None,
        beat=4,
        melody_duration_range=(0.01, 0.25),
        space_range=(0.0, 0.01),
        chord_velocity_range=(70, 90),
        melody_velocity_range=(90, 110)
    ):
        if start_time is not None:
            time = start_time
        else:
            time = self.__now_time

        pre_pitch = None
        for note_arr in chord_melody_list:
            chord_duration = np.random.uniform(low=chord_duration_range[0], high=chord_duration_range[1])
            pitch_arr = note_arr.copy()
            for i in range(note_arr.shape[0]):
                velocity = np.random.randint(low=chord_velocity_range[0], high=chord_velocity_range[1])
                note = pretty_midi.Note(velocity=velocity, pitch=note_arr[i], start=time, end=time+chord_duration)
                self.__instrument_chord.notes.append(note)

            start = time
            for i in range(beat):
                space = np.random.uniform(low=space_range[0], high=space_range[1])
                if melody_duration_range is None and beat is not None:
                    duration = np.random.uniform(low=0.01, high=(1/beat)-space)
                elif melody_duration_range is not None and beat is None:
                    duration = np.random.uniform(low=melody_duration_range[0], high=melody_duration_range[1]-space)
                elif melody_duration_range is not None and beat is not None:
                    duration = np.random.uniform(
                        low=melody_duration_range[0],
                        high=min([(1/beat), melody_duration_range[1]])-space
                    )
                else:
                    # may be unnecessary.
                    raise ValueError("The parameter of `melody_duration_range` and `beat` must be not `None`.")
                    
                end = start + duration
                if end > time + chord_duration:
                    raise ValueError("The duration of melody should match to chord.")

                velocity = np.random.randint(low=melody_velocity_range[0], high=melody_velocity_range[1])

                if i == 0:
                    np.random.shuffle(pitch_arr)
                    pitch = pitch_arr[0]
                else:
                    pitch = self.__inference_pitch(pre_pitch, pitch_arr)

                note = pretty_midi.Note(velocity=velocity, pitch=pitch, start=time, end=time+duration)
                pre_pitch = pitch
                self.__instrument_melody.notes.append(note)
                start = end + space

            time = time + chord_duration

        self.__now_time = time
        self.__pm.instruments.append(self.__instrument_chord)
        self.__pm.instruments.append(self.__instrument_melody)

    def __inference_pitch(self, pre_pitch, pitch_arr):
        test_arr = np.zeros(127)
        test_arr[pre_pitch] = 1
        test_arr = test_arr.astype(np.float64)
        self.__rbm.approximate_inferencing(
            test_arr,
            traning_count=self.__training_count, 
            r_batch_size=self.__r_batch_size
        )
        pitch_min = pitch_arr.min()
        pitch_max = pitch_arr.max()
        pitch = pitch_min + 1 + np.argmax(self.__rbm.graph.visible_activity_arr[pitch_arr.min():pitch_arr.max()])
        return pitch

    def save(self, file_path):
        self.__pm.write(file_path)
