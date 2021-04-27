# -*- coding: utf-8 -*-
from algowars.portfolio_optimization import PortfolioOptimization
import pandas as pd
import numpy as np


class VolatilityMinimization(PortfolioOptimization):
    '''
    Volatility minimization as a portfolio optimization.
    '''

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
        portfolio_std_dev_arr = np.zeros(self.portfolio_n)
        portfolio_return_arr = np.zeros(self.portfolio_n)
        weights_list = []
        for i in range(self.portfolio_n):
            weights = np.random.random(asset_n)
            weights = weights / np.sum(weights)
            weights_list.append(weights)

            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

            portfolio_std_dev_arr[i] = portfolio_std_dev
            portfolio_return_arr[i] = portfolio_return

        min_vol_idx = np.argmin(portfolio_std_dev_arr)

        random_min_vol_allocation = pd.DataFrame(
            weights_list[min_vol_idx], 
            index=ticker_list, 
            columns=['allocation']
        )
        random_min_vol_allocation.allocation = [
            round(i * 100, 2) for i in random_min_vol_allocation.allocation
        ]
        random_min_vol_allocation = random_min_vol_allocation.T

        return random_min_vol_allocation

