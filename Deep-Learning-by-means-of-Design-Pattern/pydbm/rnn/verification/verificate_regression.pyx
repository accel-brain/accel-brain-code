# -*- coding: utf-8 -*-
from logging import getLogger
from pydbm.rnn.verification.interface.verificatable_result import VerificatableResult
from pydbm.rnn.optimization.interface.optimizable_loss import OptimizableLoss
import pandas as pd
import numpy as np


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
    
    def __init__(self, optimizable_loss):
        '''
        Init.
        
        Args:
            computable_loss:    is-a `OptimizableLoss`.

        '''

        if isinstance(optimizable_loss, OptimizableLoss):
            self.__optimizable_loss = optimizable_loss
        else:
            raise TypeError()

        logger = getLogger("pydbm")
        self.__logger = logger

    def verificate(
        self,
        train_pred_arr,
        train_label_arr,
        test_pred_arr,
        test_label_arr
    ):
        '''
        Verificate result.

        Args:
            train_pred_arr:    Predicted data in training.
            train_label_arr:   Labeled data in training.
            test_pred_arr:     Predicted data in test.
            test_label_arr:    Labeled data in test.

        '''
        train_loss = self.__optimizable_loss.compute_loss(train_pred_arr, train_label_arr)
        test_loss = self.__optimizable_loss.compute_loss(test_pred_arr, test_label_arr)

        self.__logger.info("Epoch: " + str(len(self.__logs_tuple_list) + 1))

        self.__logger.info("Loss: ")
        self.__logger.info(
            "Training: " + str(train_loss) + " Test: " + str(test_loss)
        )

        self.__logs_tuple_list.append(
            (
                train_loss,
                test_loss
            )
        )

    def get_logs_df(self):
        ''' getter '''
        return pd.DataFrame(
            self.__logs_tuple_list,
            columns=[
                "train_loss",
                "test_loss"
            ]
        )

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError()
    
    logs_df = property(get_logs_df, set_readonly)
