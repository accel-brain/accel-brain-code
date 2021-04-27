# -*- coding: utf-8 -*-
from algowars.portfolio_optimization import PortfolioOptimization
import pandas as pd
import numpy as np


class SharpeRatioMaximization(PortfolioOptimization):
    '''
    Sharpe ratio maximization as a portfolio optimization.
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
        sharpe_ratio_arr = np.zeros(self.portfolio_n)
        portfolio_return_arr = np.zeros(self.portfolio_n)
        weights_list = []
        for i in range(self.portfolio_n):
            weights = np.random.random(asset_n)
            weights = weights / np.sum(weights)
            weights_list.append(weights)

            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

            sharpe_ratio_arr[i] = (portfolio_return - risk_free_rate) / (portfolio_std_dev + 1e-08)
            portfolio_return_arr[i] = portfolio_return

        max_sharpe_idx = np.argmax(sharpe_ratio_arr)
        random_max_sharpe_allocation = pd.DataFrame(
            weights_list[max_sharpe_idx],
            index=ticker_list,
            columns=['allocation']
        )
        random_max_sharpe_allocation.allocation = [
            round(i * 100, 2)for i in random_max_sharpe_allocation.allocation
        ]
        random_max_sharpe_allocation = random_max_sharpe_allocation.T

        return random_max_sharpe_allocation
