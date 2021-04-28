# -*- coding: utf-8 -*-
from algowars.technical_observer import TechnicalObserver
import numpy as np
import pandas as pd


class MultiTradeObserver(TechnicalObserver):

    def __init__(
        self, 
        technical_observer_list,
        technical_observer_weight_list=[]
    ):
        for technical_observer in technical_observer_list:
            if isinstance(technical_observer, TechnicalObserver) is False:
                raise TypeError("The type of value of `technical_observer_list` must be `TechnicalObserver`.")

        self.__technical_observer_list = technical_observer_list
        if len(technical_observer_weight_list) > 0:
            self.__technical_observer_weight_list = technical_observer_weight_list
        else:
            self.__technical_observer_weight_list = [
                1/len(technical_observer_list) for _ in range(len(technical_observer_list))
            ]

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
        weight_arr = np.zeros(len(ticker_list))
        for i in range(len(self.__technical_observer_list)):
            _weight_arr = self.__technical_observer_list[i].decide_timing(
                historical_arr,
                agents_master_arr,
                portfolio_arr,
                date_fraction,
                ticker_list,
                agent_i
            ) * self.__technical_observer_weight_list[i]
            weight_arr = weight_arr + _weight_arr

        weight_arr = weight_arr / len(self.__technical_observer_list)
        return weight_arr
