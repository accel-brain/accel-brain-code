# -*- coding: utf-8 -*-
import numpy as np


class MelodyComposer(object):
    '''
    Melody Composer.

    MIDI Note Numbers:
        C	C#	D	D#	E	F	F#	G	G#	A	A#	B
        0	1	2	3	4	5	6	7	8	9	10	11
    '''
    
    __basic_pitch_dict = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11
    }
    
    __octave = 4
    
    def get_octave(self):
        ''' getter '''
        if isinstance(self.__octave, int):
            return self.__octave
        else:
            raise TypeError()
    
    def set_octave(self, value):
        ''' setter '''
        if isinstance(octave, int):
            self.__octave = value
        else:
            raise TypeError()
    
    octave = property(get_octave, set_octave)

    def create(self, chord):
        '''
        Melody.
        
        Args:
            chord:    Chord.
        '''
        if chord == "I":
            return self.create_C()
        elif chord == "II":
            return self.create_Dm()
        elif chord == "III":
            return self.create_Em()
        elif chord == "IV":
            return self.create_F()
        elif chord == "V":
            return self.create_G()
        elif chord == "VI":
            return self.create_Am()
        elif chord == "VII":
            return self.create_Bdim()
        else:
            raise ValueError("The value of `chord` must be I - VII.")
    
    def create_C(self):
        '''
        Melody: I (G-E-C)
        '''
        return np.array([
            self.__basic_pitch_dict["G"],
            self.__basic_pitch_dict["E"],
            self.__basic_pitch_dict["C"]
        ]) + ((self.octave + 1) * 12)
    
    def create_Dm(self):
        '''
        Melody: II (A-F-D)
        '''
        return np.array([
            self.__basic_pitch_dict["A"],
            self.__basic_pitch_dict["F"],
            self.__basic_pitch_dict["D"]
        ]) + ((self.octave + 1) * 12)
    
    def create_Em(self):
        '''
        Melody: III (B-G-E)
        '''
        return np.array([
            self.__basic_pitch_dict["B"],
            self.__basic_pitch_dict["G"],
            self.__basic_pitch_dict["E"]
        ]) + ((self.octave + 1) * 12)
    
    def create_F(self):
        '''
        Melody: IV (C-A-F)
        '''
        return np.array([
            self.__basic_pitch_dict["C"],
            self.__basic_pitch_dict["A"],
            self.__basic_pitch_dict["F"]
        ]) + ((self.octave + 1) * 12)
    
    def create_G(self):
        '''
        Melody: V (D-B-G)
        '''
        return np.array([
            self.__basic_pitch_dict["D"],
            self.__basic_pitch_dict["B"],
            self.__basic_pitch_dict["G"]
        ]) + ((self.octave + 1) * 12)
    
    def create_Am(self):
        '''
        Melody: VI (E-C-A)
        '''
        return np.array([
            self.__basic_pitch_dict["E"],
            self.__basic_pitch_dict["C"],
            self.__basic_pitch_dict["A"]
        ]) + ((self.octave + 1) * 12)
    
    def create_Bdim(self):
        '''
        Melody: VII (F-D-B)
        '''
        return np.array([
            self.__basic_pitch_dict["F"],
            self.__basic_pitch_dict["D"],
            self.__basic_pitch_dict["B"]
        ]) + ((self.octave + 1) * 12)
