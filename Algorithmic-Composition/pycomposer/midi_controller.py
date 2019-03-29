# -*- coding: utf-8 -*-
import pandas as pd
import pretty_midi


class MidiController(object):
    '''
    MIDI Controller.
    '''

    def extract(self, file_path, is_drum=False):
        '''
        Extract MIDI file.
        
        Args:
            file_path:    File path of MIDI.
            is_drum:      Extract drum data or not.
            
        Returns:
            pd.DataFrame(columns=["program", "start", "end", "pitch", "velocity", "duration"])
        '''
        midi_data = pretty_midi.PrettyMIDI(file_path)
        note_tuple_list = []
        for instrument in midi_data.instruments:
            if (is_drum is False and instrument.is_drum is False) or (is_drum is True and instrument.is_drum is True):
                for note in instrument.notes:
                    note_tuple_list.append((instrument.program, note.start, note.end, note.pitch, note.velocity))
        note_df = pd.DataFrame(note_tuple_list, columns=["program", "start", "end", "pitch", "velocity"])
        note_df = note_df.sort_values(by=["program", "start", "end"])
        note_df["duration"] = note_df.end - note_df.start

        return note_df

    def save(self, file_path, note_df):
        '''
        Save MIDI file.
        
        Args:
            file_path:    File path of MIDI.
            note_df:      `pd.DataFrame` of note data.
            
        '''

        chord = pretty_midi.PrettyMIDI()
        for program in note_df.program.drop_duplicates().values.tolist():
            df = note_df[note_df.program == program]
            midi_obj = pretty_midi.Instrument(program=program)
            for i in range(df.shape[0]):
                note = pretty_midi.Note(
                    velocity=int(df.iloc[i, :]["velocity"]),
                    pitch=int(df.iloc[i, :]["pitch"]),
                    start=float(df.iloc[i, :]["start"]), 
                    end=float(df.iloc[i, :]["end"])
                )
                # Add it to our cello instrument
                midi_obj.notes.append(note)
            # Add the cello instrument to the PrettyMIDI object
            chord.instruments.append(midi_obj)
        # Write out the MIDI data
        chord.write(file_path)