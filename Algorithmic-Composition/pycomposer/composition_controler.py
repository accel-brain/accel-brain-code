# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pretty_midi
from pycomposer.chord_progression import ChordProgression
from pycomposer.melody_composer import MelodyComposer
from pycomposer.inferable_pitch import InferablePitch
from pycomposer.inferable_consonance import InferableConsonance
from pycomposer.inferable_duration import InferableDuration


class CompositionControler(object):

    # The object of `pretty_midi.PrettyMIDI`.
    __pretty_midi = None
    # The object of `ChordProgression`.
    __chord_progression = None
    # The object of `MelodyComposer`.
    __melody_composer = None
    
    __log_tuple_list = []

    def __init__(
        self,
        resolution=960,
        initial_tempo=120
    ):
        '''
        Initialize.
        
        Args:
            resolution:             Resolution of the MIDI data, when no file is provided.
            initial_tempo:          Initial tempo for the MIDI data, when no file is provided.
        '''
        self.reset(
            resolution,
            initial_tempo
        )
        self.__chord_progression = ChordProgression()
        self.__melody_composer = MelodyComposer()

    def reset(
        self,
        resolution=960,
        initial_tempo=120
    ):
        '''
        Reset.
        
        Args:
            resolution:             Resolution of the MIDI data, when no file is provided.
            initial_tempo:          Initial tempo for the MIDI data, when no file is provided.

        '''
        self.__pretty_midi = pretty_midi.PrettyMIDI(resolution=resolution, initial_tempo=initial_tempo)
        self.__log_tuple_list = []

    def create_chord_list(
        self,
        octave,
        first_chord="I",
        total_measure_n=8
    ):
        '''
        Create chord progression.
        
        Args:
            octave:             octave.
            first_chord:        The string of Diatonic code. (I, II, III, IV, V, VI, VII)
            total_measure_n:    The length of measures.
        
        Retruns:
            The list of `np.ndarray` that contains the string of Diatonic code.
        '''
        chord_list = [self.__melody_composer.create(first_chord)]
        for i in range(total_measure_n):
            chord = self.__chord_progression.progression(state=first_chord, octave=octave)
            chord_list.append(self.__melody_composer.create(chord))

        return chord_list
    
    def compose_chord(
        self,
        chord_list,
        metronome_time=60, 
        start_measure_n=0,
        measure_n=4,
        beat_n=4,
        chord_instrument_num=39,
        chord_velocity_range=(70, 90),
    ):
        '''
        Compose chords.
        
        Args:
            chord_list:             The list of `np.ndarray` that contains the string of Diatonic code.
            metronome_time:         Metronome time.
            start_measure_n:        The timing of the beginning of the measure.
            beat_n:                 The number of beats.
            chord_instrument_num:   MIDI program number (instrument index), in [0, 127].
            chord_velocity_range:   The tuple of chord velocity in MIDI.
                                    The form of tuple is (low velocity, high veloicty).
                                    The value of velocity is determined by `np.random.randint`.

        @TODO(chimera0):    Reconsider specification of velocity.

        '''
        instrument_chord = pretty_midi.Instrument(chord_instrument_num)

        chord_time = 0.0
        for measure in range(len(chord_list)):
            pitch_arr = chord_list[measure]
            measure = measure + 1 + start_measure_n
            start = chord_time
            end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat_n))
            velocity = np.random.randint(low=chord_velocity_range[0], high=chord_velocity_range[1])
            for i in range(pitch_arr.shape[0]):
                note = pretty_midi.Note(
                    velocity=velocity, 
                    pitch=pitch_arr[i], 
                    start=start, 
                    end=end
                )
                instrument_chord.notes.append(note)
                self.__log_tuple_list.append((
                    start,
                    end,
                    pitch_arr[i],
                    velocity
                ))
            chord_time = end

        self.__pretty_midi.instruments.append(instrument_chord)
    
    def compose_melody(
        self,
        inferable_pitch,
        inferable_consonance,
        inferable_duration,
        chord_list,
        total_measure_n=40,
        measure_n=4,
        start_measure_n=0,
        beat_n=4,
        metronome_time=60,
        melody_instrument_num=0,
        melody_velocity_range=(90, 110)
    ):
        '''
        Compose melody.
        
        Args:
            inferable_pitch:            The object of `InferablePitch`.
            inferable_consonance:       The object of `InferableConsonance`.
            inferable_duration:         The object of `InferableDuration`.
            chord_list:                 The list of `np.ndarray` that contains the string of Diatonic code.
            total_measure_n:            The length of measures.
            measure_n:                  The number of measures.
            start_measure_n:            The timing of the beginning of the measure.
            beat_n:                     The number of beats.
            metronome_time:             Metronome time.
            melody_instrument_num:      MIDI program number (instrument index), in [0, 127].
            melody_velocity_range:      The tuple of melody velocity in MIDI.
                                        The form of tuple is (low velocity, high veloicty).
                                        The value of velocity is determined by `np.random.randint`.

        @TODO(chimera0):    Reconsider specification of velocity.

        '''

        if isinstance(inferable_pitch, InferablePitch) is False:
            raise TypeError("The type of `inferable_pitch` must be `InferablePitch`.")
        if isinstance(inferable_consonance, InferableConsonance) is False:
            raise TypeError("The type of `inferable_consonance` must be `InferableConsonance`.")
        if isinstance(inferable_duration, InferableDuration) is False and inferable_duration is not None:
            raise TypeError("The type of `inferable_duration` must be `InferableDuration` or `None`.")

        instrument_melody = pretty_midi.Instrument(melody_instrument_num)

        chord_time = 0.0
        melody_time = 0.0
        for measure in range(total_measure_n):
            pitch_arr = chord_list[measure]
            duration_arr = None
            if inferable_duration is not None:
                duration_arr = inferable_duration.inference(
                    chord_time,
                    measure,
                    metronome_time,
                    measure_n,
                    beat_n,
                    total_measure_n
                )
                chord_end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat_n))
                chord_time = chord_end

            for beat in range(beat_n):
                beat = beat + 1
                start = melody_time
                if duration_arr is None:
                    end = (((60/metronome_time) * measure_n * (measure))) + ((60/metronome_time) * (beat))
                else:
                    end = start + duration_arr[beat - 1]

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
                instrument_melody.notes.append(note)
                self.__log_tuple_list.append((
                    start,
                    end,
                    pitch,
                    velocity
                ))

                pre_pitch = pitch
                melody_time = end

        self.__pretty_midi.instruments.append(instrument_melody)

    def save(self, file_path):
        '''
        Save MIDI file.
        
        Args:
            file_path:    Saved file path.
        '''
        self.__pretty_midi.write(file_path)

    def export_df(self):
        '''
        Return `pd.DataFrame`.
        
        Returns:
            Data Frame.
        '''
        return pd.DataFrame(self.__log_tuple_list, columns=["start", "end", "pitch", "velocity"])
