# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.verification.interface.verificatable_result import VerificatableResult
from pydbm.loss.interface.computable_loss import ComputableLoss
import pandas as pd
import numpy as np


class VerificateSoftmax(VerificatableResult):
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
    
    def __init__(self):
        '''
        Init.
        
        Args:
            computable_loss:    is-a `OptimizableLoss`.
        '''
        logger = getLogger("pydbm")
        self.__logger = logger

    def verificate(
        self,
        computable_loss,
        train_pred_arr,
        train_label_arr,
        test_pred_arr,
        test_label_arr,
        train_penalty=0.0,
        test_penalty=0.0
    ):
        '''
        Verificate result.

        Args:
            computable_loss:   is-a `ComputableLoss`.
            train_pred_arr:    Predicted data in training.
            train_label_arr:   Labeled data in training.
            test_pred_arr:     Predicted data in test.
            test_label_arr:    Labeled data in test.
            train_penalty:     Sum of penalty terms in training.
            test_penalty:      Sum of penalty terms in test.

        '''
        if isinstance(computable_loss, ComputableLoss) is False:
            raise TypeError()
        
        train_loss = computable_loss.compute_loss(train_pred_arr, train_label_arr)
        test_loss = computable_loss.compute_loss(test_pred_arr, test_label_arr)

        train_loss = train_loss + train_penalty
        test_loss = test_loss + test_penalty

        self.__logger.info("Epoch: " + str(len(self.__logs_tuple_list) + 1))

        self.__logger.info("Loss: ")
        self.__logger.info(
            "Training: " + str(train_loss) + " Test: " + str(test_loss)
        )

        self.__logs_tuple_list.append(
            (
                train_loss,
                test_loss,
            )
        )

    def get_logs_df(self):
        ''' getter '''
        return pd.DataFrame(
            self.__logs_tuple_list,
            columns=[
                "train_loss",
                "test_loss",
            ]
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    logs_df = property(get_logs_df, set_readonly)