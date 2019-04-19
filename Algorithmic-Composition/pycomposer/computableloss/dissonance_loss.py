# -*- coding: utf-8 -*-
import numpy as np
from pydbm.loss.interface.computable_loss import ComputableLoss


class DissonanceLoss(ComputableLoss):
    '''
    Cost function.
    
    This function treats a sound sequences containing *dissonance* as a kind of *loss*.
    '''

    # `list` of score of frequency ratio.
    __freq_ratio_score_list = [
        0.693,
        3.434,
        2.833,
        2.398,
        2.197,
        1.946,
        4.344,
        1.609,
        2.565,
        2.079,
        3.219,
        3.135,
        1.099
    ]
    
    def __init__(self, computable_loss, freq_ratio_score_list=None, min_pitch=60):
        '''
        Init.
        
        Args:
            computable_loss:            is-a `ComputableLoss`.
            freq_ratio_score_list:      `list` of score of frequency ratio.
                                        In default, the values will be computed as the logarithm of the sum of intergers(Schellenberg, E. G., et al., 1994).

            min_pitch:                  The minimum of pitch.
        '''
        if isinstance(computable_loss, ComputableLoss) is False:
            raise TypeError()
        self.__computable_loss = computable_loss
        if freq_ratio_score_list is not None:
            self.__freq_ratio_score_list = freq_ratio_score_list
        
        self.__min_pitch = min_pitch
    
    def compute_loss(self, pred_arr, labeled_arr, axis=None):
        '''
        Return of result from this Cost function.

        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            axis:           Axis or axes along which the losses are computed.
                            The default is to compute the losses of the flattened array.

        Returns:
            Cost.
        '''
        loss = self.__computable_loss.compute_loss(pred_arr, labeled_arr, axis)
        dissonance_loss_arr = self.__compute_dissonance(labeled_arr) - self.__compute_dissonance(pred_arr)
        loss = loss + dissonance_loss_arr.mean()
        return loss

    def compute_delta(self, pred_arr, labeled_arr, delta_output=1):
        '''
        Backward delta.
        
        Args:
            pred_arr:       Predicted data.
            labeled_arr:    Labeled data.
            delta_output:   Delta.

        Returns:
            Delta.
        '''
        delta_arr = self.__computable_loss.compute_delta(pred_arr, labeled_arr, delta_output)
        dissonance_loss_arr = self.__compute_dissonance(labeled_arr) - self.__compute_dissonance(pred_arr)
        delta_arr = delta_arr + dissonance_loss_arr.reshape((
            dissonance_loss_arr.shape[0], 
            dissonance_loss_arr.shape[1], 
            1,
            1
        ))
        return delta_arr

    def __compute_dissonance(self, arr):
        loss_arr = np.zeros((arr.shape[0], arr.shape[1], 1))
        for batch in range(arr.shape[0]):
            for channel in range(arr.shape[1]):
                y_arr, x_arr = np.where(arr[batch, channel] == 1)
                loss_list = []
                for i in range(arr[batch, channel].shape[0]):
                    loss = 0.0
                    pitch_arr = x_arr[y_arr == i]
                    if pitch_arr.shape[0] > 0:
                        pitch_arr = pitch_arr + self.__min_pitch
                        pitch_arr = pitch_arr % 12

                        for j in range(pitch_arr.shape[0]):
                            for k in range(pitch_arr.shape[0]):
                                if j == k:
                                    continue
                            interval_key = abs(pitch_arr[j] - pitch_arr[k])
                            loss += self.__freq_ratio_score_list[interval_key]
                    loss_list.append(loss)
                loss_arr[batch, channel] = sum(loss_list)

        loss_arr = np.nan_to_num(loss_arr)
        return loss_arr
