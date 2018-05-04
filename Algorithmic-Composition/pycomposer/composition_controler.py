# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pretty_midi
from pycomposer.chord_progression import ChordProgression
from pycomposer.melody_composer import MelodyComposer
from pycomposer.inferable_pitch import InferablePitch
from pycomposer.inferable_consonance import InferableConsonance


class CompositionControler(object):

    # The object of `pretty_midi.PrettyMIDI`.
    __pretty_midi = None
    # The object of `pretty_midi.Instrument`.
    __instrument_chord = None
    # The object of `pretty_midi.Instrument`.
    __instrument_melody = None
    # The object of `ChordProgression`.
    __chord_progression = None
    # The object of `MelodyComposer`.
    __melody_composer = None

    def __init__(
        self,
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

    def reset(
        self,
        resolution=960,
        initial_tempo=120,
        chord_instrument_num=33,
        melody_instrument_num=1
    ):
        self.__pretty_midi = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
        self.__instrument_chord = pretty_midi.Instrument(chord_instrument_num)
        self.__instrument_melody = pretty_midi.Instrument(melody_instrument_num)

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
        inferable_pitch,
        inferable_consonance,
        chord_melody_list,
        measure_n=4,
        start_measure_n=0,
        beat_n=4,
        metronome_time=60,
        chord_velocity_range=(70, 90),
        melody_velocity_range=(90, 110)
    ):
        if isinstance(inferable_pitch, InferablePitch) is False:
            raise TypeError("The type of `inferable_pitch` must be `InferablePitch`.")
        if isinstance(inferable_consonance, InferableConsonance) is False:
            raise TypeError("The type of `inferable_consonance` must be `InferableConsonance`.")

        chord_time = 0.0
        for measure in range(len(chord_melody_list)):
            pitch_arr = chord_melody_list[measure]
            measure = measure + 1 + start_measure_n
            start = chord_time
            end = (((60/metronome_time) * 4 * (measure - 1))) + ((60/metronome_time) * (4 - 1))
            velocity = np.random.randint(low=chord_velocity_range[0], high=chord_velocity_range[1])
            for i in range(pitch_arr.shape[0]):
                note = pretty_midi.Note(velocity=velocity, pitch=pitch_arr[i], start=start, end=end)
                self.__instrument_chord.notes.append(note)
            chord_time = end

        melody_time = 0.0
        for measure in range(measure_n):
            pitch_arr = chord_melody_list[measure]
            measure = measure + 1 + start_measure_n
            for beat in range(beat_n):
                beat = beat + 1
                start = melody_time
                end = (((60/metronome_time) * 4 * (measure - 1))) + ((60/metronome_time) * (beat - 1))

                if beat == 1:
                    np.random.shuffle(pitch_arr)
                    pitch = pitch_arr[0]
                else:
                    consonance_pitch_arr = inferable_consonance.inference(pre_pitch, limit=5)
                    if consonance_pitch_arr.shape[0]:
                        _pitch_arr = consonance_pitch_arr
                    else:
                        _pitch_arr = pitch_arr
                    pitch = inferable_pitch.inferance(pre_pitch, _pitch_arr)

                velocity = np.random.randint(low=melody_velocity_range[0], high=melody_velocity_range[1])

                note = pretty_midi.Note(
                    velocity=velocity, 
                    pitch=pitch, 
                    start=start, 
                    end=end
                )
                self.__instrument_melody.notes.append(note)
                pre_pitch = pitch
                melody_time = end

        self.__pretty_midi.instruments.append(self.__instrument_chord)
        self.__pretty_midi.instruments.append(self.__instrument_melody)

    def save(self, file_path):
        self.__pretty_midi.write(file_path)
