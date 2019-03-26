# -*- coding: utf-8 -*-
from pygan.generativemodel.lstm_model import LSTMModel


class MidiLSTMModel(LSTMModel):
    '''
    LSTM as a Generator for MIDI format data.
    '''

    def draw(self):
        '''
        Draws samples from the `fake` distribution.

        Returns:
            `np.ndarray` of samples.
        '''
        arr = super().draw()
        if arr.max() > arr.min():
            arr = (arr - arr.min()) / (arr.max() - arr.min())
        return arr

    def learn(self, grad_arr):
        '''
        Update this Discriminator by ascending its stochastic gradient.

        Args:
            grad_arr:   `np.ndarray` of gradients.
        
        '''
        if grad_arr.max() > grad_arr.min():
            grad_arr = (grad_arr - grad_arr.min()) / (grad_arr.max() - grad_arr.min())
        super().learn(grad_arr)
