# -*- coding: utf-8 -*-
from algowars.technical_observer import TechnicalObserver
import numpy as np
import pandas as pd


class MACDObserver(TechnicalObserver):

    def __init__(
        self, 
        time_window=None,
        short_term=12,
        long_term=26,
        signal_term=9
    ):
        self.__time_window = time_window
        self.__short_term = short_term
        self.__long_term = long_term
        self.__signal_term = signal_term

        max_term_list = [
            short_term,
            long_term,
            signal_term
        ]
        if self.__time_window is not None:
            max_term_list.append(self.__time_window)

        self.__max_term = max(max_term_list)

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
                signal = self.__compute_macd(close_arr)
                weight_arr[ticker_i] = signal * weight_arr[ticker_i]

            if weight_arr.max() != weight_arr.min():
                weight_arr = (weight_arr - weight_arr.min()) / (weight_arr.max() - weight_arr.min())
                weight_arr = weight_arr * p

        return weight_arr

    def __compute_macd(self, close_arr):
        close_df = pd.DataFrame(close_arr, columns=["close"])
        close_df["ema_short"] = close_df["close"].ewm(span=self.__short_term).mean()
        close_df["ema_long"] = close_df["close"].ewm(span=self.__long_term).mean()
        close_df["macd"] = close_df["ema_short"] - close_df["ema_long"]
        close_df["signal"] = close_df["macd"].ewm(span=self.__signal_term).mean()

        signal = 0.0
        if close_df.macd.iloc[-2] > close_df.signal.iloc[-2] and close_df.macd.iloc[-1] < close_df.signal.iloc[-1]:
            signal = 1.0
        elif close_df.macd.iloc[-2] < close_df.signal.iloc[-2] and close_df.macd.iloc[-1] > close_df.signal.iloc[-1]:
            signal = -1.0

        return signal
