# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class TechnicalObserver(metaclass=ABCMeta):
    '''
    The interface to decide timing of trades.
    '''

    @abstractmethod
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
        raise NotImplementedError()
