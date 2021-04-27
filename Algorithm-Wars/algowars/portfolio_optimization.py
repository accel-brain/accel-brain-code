# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod


class PortfolioOptimization(metaclass=ABCMeta):
    '''
    The abstract class for a portfolio optimization.
    '''

    # The number of portfolio.
    __portfolio_n = 100

    def get_portfolio_n(self):
        ''' getter for the number of portfolio.'''
        return self.__portfolio_n
    
    def set_portfolio_n(self, value):
        ''' setter for the number of portfolio. '''
        self.__portfolio_n = value

    portfolio_n = property(get_portfolio_n, set_portfolio_n)

    # Now opitmization policy.
    __opt_policy = "max_sharpe_ratio"

    def get_opt_policy(self):
        ''' getter for now opitmization policy.'''
        return self.__opt_policy

    def set_opt_policy(self, value):
        ''' setter for now opitmization policy.'''
        if value not in self.opt_policy_list:
            raise ValueError("The value of `opt_policy` must be " + ", ".join(self.opt_policy_list) + ".")
        self.__opt_policy = value

    opt_policy = property(get_opt_policy, set_opt_policy)

    # `list` of opitmization policy.
    __opt_policy_list = ["max_sharpe_ratio", "min_vol"]

    def get_opt_policy_list(self):
        ''' getter `list` of opitmization policy. '''
        return self.__opt_policy_list

    def set_opt_policy_list(self, value):
        ''' setter `list` of opitmization policy. '''
        self.__opt_policy_list = value

    opt_policy_list = property(get_opt_policy_list, set_opt_policy_list)

    __cov_matrix = None

    def optimize(self, histroical_arr, risk_free_rate=0.3):
        '''
        Optimize.

        Args:
            histroical_arr:     rank-1 4D `np.ndarray` of histrocal data.
                                    - val1: Date key.
                                    - val2: Values(unit prices).
                                    - val3: Volumes.
                                    - val4: Ticker.
                                This array must be sorted by `val1` and unique key of each stock

            risk_free_rate:     `float` of risk free rate.

        Returns:
            'pd.DataFrame'.
        '''
        if histroical_arr.shape[1] == 5:
            histroical_df = pd.DataFrame(
                histroical_arr,
                columns=["date", "value", "volume", "ticker", "_date"]
            )
        else:
            histroical_df = pd.DataFrame(
                histroical_arr,
                columns=["date", "value", "volume", "ticker"]
            )

        histroical_df = histroical_df[["date", "value", "ticker"]]

        asset_n = histroical_df.ticker.drop_duplicates().shape[0]

        df = histroical_df.set_index('date')
        pivot_df = df.pivot(columns='ticker')
        pivot_df = pivot_df.fillna(0.0)

        pivot_df.columns = [col[1] for col in pivot_df.columns]

        returns_df = pivot_df.pct_change().fillna(0)
        returns_df = returns_df.replace(np.inf, np.nan).fillna(0.0)
        mean_returns = returns_df.mean()
        if returns_df.shape[0] > 1:
            cov_matrix = returns_df.cov().fillna(0)
            self.__cov_matrix = cov_matrix
        else:
            if self.__cov_matrix is None:
                cov_matrix = pd.DataFrame(np.zeros(asset_n))
            else:
                cov_matrix = self.__cov_matrix

        return self.select(
            asset_n, 
            mean_returns, 
            cov_matrix, 
            risk_free_rate,
            pivot_df.columns
        )

    @abstractmethod
    def select(
        self, 
        asset_n, 
        mean_returns, 
        cov_matrix, 
        risk_free_rate,
        ticker_list
    ):
        '''
        Do the portfolio selection.

        Args:
            asset_n:        `int` of the number of assets.
            mean_returns:   `float` of average returns.
            cov_matrix:     `pd.DataFrame` of covariance matrix.
            risk_free_rate: `float` of risk free rate.
            ticker_list:    `list` of ticker symbol.
        
        Returns:
            `pd.DataFrame`.
        '''
        raise NotADirectoryError()
