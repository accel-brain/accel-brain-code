# -*- coding: utf-8 -*-
from algowars.technical_observer import TechnicalObserver
import numpy as np
import pandas as pd


class BollingerBandObserver(TechnicalObserver):

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
        tickerごとに増減させる割合を出力するように改修する。
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
                signal = self.__compute_ticker_signal(close_arr)
                weight_arr[ticker_i] = signal * weight_arr[ticker_i]

        if weight_arr.max() != weight_arr.min():
            weight_arr = (weight_arr - weight_arr.min()) / (weight_arr.max() - weight_arr.min())
            weight_arr = weight_arr * p

        return weight_arr

    def __compute_ticker_signal(self, close_arr):
        if close_arr.shape[0] > 0:
            mu = close_arr[-self.__time_window:].mean()
            sigma = close_arr[-self.__time_window:].std()

            if close_arr[-1] > (mu + (2 * sigma)):
                signal = close_arr[-1] - (mu + (2 * sigma))
            elif close_arr[-1] < (mu - (2 * sigma)):
                signal = close_arr[-1] - (mu - (2 * sigma))
            else:
                signal = 1.0
        else:
            signal = 1.0

        return signal
