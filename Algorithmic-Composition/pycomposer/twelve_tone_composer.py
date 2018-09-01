# -*- coding: utf-8 -*-
import numpy as np


class TwelveToneComposer(object):
    '''
    Twelve Tone Composer.
    
    Create 12 tone row and compose with the transposition, retrograde, and inversion.
    '''
    
    def generate(self, rows_num=10, start_octave=6, octave_range=(5, 7)):
        '''
        Composition.

        Args:
            rows_num:        The number of twelve tone rows.
            start_octave:    Octave in first row.
            octave_range:    `Tuple` of range of octave. The shape is: (Min, Max).
        
        Returns:
            `np.ndarray` of twelve tone rows.
        '''
        min_octave, max_octave = octave_range
        if min_octave > start_octave or max_octave < start_octave:
            raise ValueError("`start_octave` is invalid in relation to `octave_range`.")

        self.__min_pitch = 12 * (int(min_octave)+1)
        self.__max_pitch = 12 * (int(max_octave)+1)

        twelve_tones_arr = np.empty((rows_num, 12))
        for i in range(rows_num):
            if i == 0:
                twelve_tones_arr[i] = self.compose(start_octave)
            else:
                flag = np.random.randint(low=0, high=4)
                if flag == 0:
                    high = self.__max_pitch - twelve_tones_arr[i-1].max()
                    low = self.__min_pitch - twelve_tones_arr[i-1].min()
                    add_pitch = np.random.randint(low=low, high=high)
                    if twelve_tones_arr[i-1][twelve_tones_arr[i-1] + add_pitch > self.__max_pitch].shape[0]:
                        add_pitch = self.__max_pitch - twelve_tones_arr[i-1].max()
                    if twelve_tones_arr[i-1][twelve_tones_arr[i-1] + add_pitch <= self.__min_pitch].shape[0]:
                        add_pitch = self.__min_pitch + 1 - twelve_tones_arr[i-1].min()

                    twelve_tones_arr[i] = self.transpose(twelve_tones_arr[i-1], add_pitch)

                elif flag == 1:
                    twelve_tones_arr[i] = self.retrograde(twelve_tones_arr[i-1])
                elif flag == 2:
                    axis = np.random.randint(low=0, high=twelve_tones_arr[i-1].shape[0])
                    twelve_tones_arr[i] = self.inversion(twelve_tones_arr[i-1], axis)
                elif flag == 3:
                    axis = np.random.randint(low=0, high=twelve_tones_arr[i-1].shape[0])
                    twelve_tones_arr[i] = self.inverse_retrograde(twelve_tones_arr[i-1], axis)
                else:
                    twelve_tones_arr[i] = self.compose(start_octave)

        return twelve_tones_arr.ravel()

    def compose(self, octave):
        '''
        Composition.

        Args:
            octave:    Octave notation.
        
        Returns:
            twelve tone row.
        '''
        arr = np.arange(12)
        if octave >= -1 and octave <= 9:
            arr += 12 * (int(octave)+1)
        else:
            raise ValueError("The int of octave must be `-1` - `9` but the value is " + str(octave))

        if octave == 9:
            arr = arr[arr <= 127]

        np.random.shuffle(arr)
        return arr

    def transpose(self, twelve_tone_row, add_pitch):
        '''
        Transposition.
        
        Args:
            twelve_tone_row:    Twelve tone row.
            add_pitch:          Value of pitch to be added for transposition.

        Returns:
            Twelve tone row.

        Exceptions:
            ValueError:   In the case that one of the value of transposed `twelve_tone_row` is not satisfied; 0 - 127.
        '''
        twelve_tone_row += int(add_pitch)
        if twelve_tone_row[twelve_tone_row < 0].shape[0] > 0 or twelve_tone_row[twelve_tone_row > 127].shape[0] > 0:
            raise ValueError("The value of twelve tone row must be `0` - `127`.")
        return twelve_tone_row

    def retrograde(self, twelve_tone_row):
        '''
        Retrograde.
        
        Args:
            twelve_tone_row:    Twelve tone row.
        
        Returns:
            Twelve tone row.
        '''
        return twelve_tone_row[::-1]
    
    def inversion(self, twelve_tone_row, axis):
        '''
        Inversion.
        
        Args:
            twelve_tone_row:    Twelve tone row.
            axis:               Axis for inversion.

        Returns:
            Twelve tone row.
        '''
        inversion_arr = -2 * (twelve_tone_row - twelve_tone_row[axis])
        twelve_tone_row = twelve_tone_row + inversion_arr
        return twelve_tone_row
    
    def inverse_retrograde(self, twelve_tone_row, axis):
        '''
        Inversion and Retrograde.
        
        Args:
            twelve_tone_row:    Twelve tone row.
            axis:               Axis for inversion.

        Returns:
            Twelve tone row.
        '''
        return self.retrograde(self.inversion(twelve_tone_row, axis))
