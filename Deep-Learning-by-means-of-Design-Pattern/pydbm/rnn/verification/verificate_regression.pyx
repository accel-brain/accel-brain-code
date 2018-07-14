# -*- coding: utf-8 -*-
from logging import getLogger
import numpy as np
cimport numpy as np
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
from pydbm.rnn.loss.interface.computable_loss import ComputableLoss
import pandas as pd


class VerificateRegression(VerificatableResult):
    '''
    Verification of softmax result.
    '''

    # Logs of accuracy.
    __logs_tuple_list = []
    __total_train_n = 0
    __total_test_n = 0
    __total_train_match_n = 0
    __total_test_match_n = 0
    
    # Logger.
    __logger = None
    
    def __init__(self, computable_loss):
        '''
        Init.
        
        Args:
            computable_loss:    is-a `OptimizableLoss`.

        '''

        if isinstance(computable_loss, ComputableLoss):
            self.__computable_loss = computable_loss
        else:
            raise TypeError()

        logger = getLogger("pydbm")
        self.__logger = logger

    def verificate(
        self,
        np.ndarray train_pred_arr,
        np.ndarray train_label_arr,
        np.ndarray test_pred_arr,
        np.ndarray test_label_arr
    ):
        '''
        Verificate result.

        Args:
            train_pred_arr:    Predicted data in training.
            train_label_arr:   Labeled data in training.
            test_pred_arr:     Predicted data in test.
            test_label_arr:    Labeled data in test.

        '''
        train_loss = self.__computable_loss.compute_loss(train_pred_arr, train_label_arr)
        test_loss = self.__computable_loss.compute_loss(test_pred_arr, test_label_arr)
        train_r2 = self.__r2_score(train_pred_arr, train_label_arr)
        test_r2 = self.__r2_score(test_pred_arr, test_label_arr)

        self.__logger.debug("Epoch: " + str(len(self.__logs_tuple_list) + 1))

        self.__logger.debug("Loss: ")
        self.__logger.debug(
            "Training: " + str(train_loss) + " Test: " + str(test_loss)
        )
        self.__logger.debug("R2 score: ")
        self.__logger.debug(
            "Training: " + str(train_r2) + " Test: " + str(test_r2)
        )
        
        self.__logs_tuple_list.append(
            (
                train_loss,
                test_loss,
                train_r2,
                test_r2
            )
        )

    def __r2_score(self, np.ndarray pred_arr, np.ndarray label_arr):
        '''
        R2.
        
        Args:
            pred_arr:    Predicted array.
            label_arr:   Labeled array.
        
        Returns:
            R2 score.
        '''
        return 1 - (np.sum(np.power(label_arr - pred_arr, 2)) / np.sum(np.power(label_arr - np.mean(label_arr), 2)))

    def get_logs_df(self):
        ''' getter '''
        return pd.DataFrame(
            self.__logs_tuple_list,
            columns=[
                "train_loss",
                "test_loss",
                "train_r2",
                "test_r2"
            ]
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    logs_df = property(get_logs_df, set_readonly)
