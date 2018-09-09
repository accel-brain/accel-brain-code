# -*- coding: utf-8 -*-
from pycomposer.extractable_melody import ExtractableMelody


class SkylineExtractor(ExtractableMelody):
    '''
    Melody extractor based on Skyline algorithm.
    
    Assume that $M$ is a MIDI file composed of channels. Formally, 
    
    $$M = \{c_1, c_2, ..., c_i\} \ (1 < i < 16)$$
    
    where $c_i$ is a set of channel, containing $k$ notes. Mathematically, 
    
    $$c_i = \{n_{i1}, n_{i2}, ..., n_{ik\}$$
    
    where $n_{ij} has three important properties: `pitch`, `onset` time and `offset` time,
    $p_{ij}$, $s_{ij}$, $e_{ij}$ respectively. we define a note as a set as follows: 
    
    $$n_{ij} = \{p_{ij}, s_{ij}, e_{ij}\}$$
    
    where $0 < p_{ij} <= 128$.
    
    Each note is sorted by $s_{ij}$.
    
    References:
        Ozcan, G., Isikhan, C., & Alpkocak, A. (2005, December). Melody extraction on MIDI music files. In Multimedia, Seventh IEEE International Symposium on (pp. 8-pp). Ieee.
    '''

    def extract(self, midi_arr):
        '''
        Extract melody.
        
        Args:
            midi_arr:       `np.ndarray` of a MIDI file composed of channels. 
                            The shape is: (
                                `The number of channels`,
                                `The number of notes`, 
                                `Pitch`,
                                `start offset time`,
                                `end offset time`
                            )
        
        Returns:
            `np.ndarray` of melody.
        '''
        for i in range(midi_arr.shape[0]):
            # $n_{ij}$
            for j in range(midi_arr[i].shape[0] - 1):
                k = j + 1
                while midi_arr[i][j][1] == midi_arr[i][k][1]:
                    # pitch
                    if midi_arr[i][j][0] < midi_arr[i][k][0]:
                        # To eliminate $p_{ij}$
                        midi_arr[i][j][0] = -1
                        j = k
                    else:
                        # To eliminate $p_{ik}$
                        midi_arr[i][k][0] = -1
                        k = k + 1
                    # $e_{ij} > s_{ik}$ then $e_{ij} = s_{ik}$
                    try:
                        if midi_arr[i][j][2] > midi_arr[i][k][1]:
                            midi_arr[i][j][2] = midi_arr[i][k][1]
                    except IndexError:
                        break
                j = k

        # Execution of eliminations.
        midi_arr = midi_arr[midi_arr[:, :, 0] != -1]
        return midi_arr
