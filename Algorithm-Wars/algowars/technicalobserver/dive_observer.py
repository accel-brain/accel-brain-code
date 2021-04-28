# -*- coding: utf-8 -*-
from algowars.information import entropy
from algowars.information import joint_entropy
from algowars.information import conditional_entropy
from algowars.information import thermodynamic_depth
from algowars.information import thermodynamic_dive

from algowars.technical_observer import TechnicalObserver

import numpy as np
import pandas as pd


class DiveObserver(TechnicalObserver):

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
            Tuple data.
            - `bool`. Trade or not.
            - `float`. Probability.
        '''
        if self.__time_window is None:
            self.__time_window = date_fraction

        weight_arr = np.ones(len(ticker_list))
        if historical_arr.shape[0] > self.__time_window * 2 * len(ticker_list):
            for ticker_i in range(len(ticker_list)):
                ticker = ticker_list[ticker_i]
                close_arr = historical_arr[historical_arr[:, 3] == ticker][:, 1].reshape(-1, 1)

                close_arr = close_arr[-self.__time_window*2:]
                close_arr = (close_arr[1:] > close_arr[:1]).astype(int)

                dive = thermodynamic_dive(S=close_arr)
                dive_weight = np.exp(dive)

                weight_arr[ticker_i] = weight_arr[ticker_i] * dive_weight

            if weight_arr.max() != weight_arr.min():
                weight_arr = (weight_arr - weight_arr.min()) / (weight_arr.max() - weight_arr.min())
                weight_arr = weight_arr * p

        return rebalance_flag, p
