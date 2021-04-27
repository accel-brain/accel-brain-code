# -*- coding: utf-8 -*-
from algowars.technical_observer import TechnicalObserver
import numpy as np
import pandas as pd


class ContrarianRSIObserver(TechnicalObserver):

    def __init__(
        self, 
        time_window=None,
    ):
        self.__time_window = time_window

    def decide_timing(
        self,
        historical_arr,
        agents_master_arr,
        portfolio_arr,
        date_fraction,
        ticker_list,
        agent_i
    ):
        '''
        Decide timing of trades.

        Args:
            historical_arr:     `np.ndarray` of historical data.
            agents_master_arr:  `np.ndarray` of agents master data.
            portfolio_arr:      `np.ndarray` of now agent's portfolio.
            date_fraction:      `int` of date fraction.
            ticker_list:        `list` of ticker symbols.
            agent_i:            `int` of agent's index.

        Returns:
            `np.array` of portfolio weights.
        '''
        if self.__time_window is None:
            self.__time_window = date_fraction

        weight_arr = np.ones(len(ticker_list))
        if historical_arr.shape[0] > self.__time_window * 2 * len(ticker_list):
            p = agents_master_arr[agent_i][3]
            signal_list = []
            for ticker_i in range(len(ticker_list)):
                ticker = ticker_list[ticker_i]
                close_arr = historical_arr[historical_arr[:, 3] == ticker][:, 1].reshape(-1, 1)
                signal = self.__compute_rsi(close_arr)
                weight_arr[ticker_i] = signal * weight_arr[ticker_i]

        weight_arr = 1 / weight_arr
        if weight_arr.max() != weight_arr.min():
            weight_arr = (weight_arr - weight_arr.min()) / (weight_arr.max() - weight_arr.min())
            weight_arr = weight_arr * p

        return weight_arr

    def __compute_rsi(self, close_arr):
        with np.errstate(invalid='ignore'):
            if close_arr.shape[0] > 2:
                diff_arr = close_arr[1:] - close_arr[:-1]
                diff_arr = np.nan_to_num(diff_arr)
                positive = np.nansum(diff_arr[diff_arr > 0])
                negative = np.nansum(diff_arr[diff_arr < 0]) * -1

                try:
                    if positive + negative != 0:
                        rsi = np.mean(positive / (positive + negative))
                    else:
                        rsi = 0.5
                except:
                    rsi = 0.5
            else:
                rsi = 0.5

        return rsi
