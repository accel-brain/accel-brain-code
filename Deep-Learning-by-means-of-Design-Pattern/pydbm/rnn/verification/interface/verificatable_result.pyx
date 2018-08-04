# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class VerificatableResult(metaclass=ABCMeta):
    '''
    The interface for verification of result.
    '''

    @abstractmethod
    def verificate(
        self,
        computable_loss,
        train_pred_arr,
        train_label_arr,
        test_pred_arr,
        test_label_arr
    ):
        '''
        Verificate result.

        Args:
            computable_loss:   is-a `ComputableLoss`.
            train_pred_arr:    Predicted data in training.
            train_label_arr:   Labeled data in training.
            test_pred_arr:     Predicted data in test.
            test_label_arr:    Labeled data in test.
        
        '''
        raise NotImplementedError()