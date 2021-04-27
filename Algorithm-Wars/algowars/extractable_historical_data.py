# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ExtractableHistoricalData(metaclass=ABCMeta):
    '''
    The interface to load, save, and extract the historical data.
    '''

    @abstractmethod
    def extract(self, start_date, end_date, ticker_list):
        '''
        Extract histroical data from local csv file.

        Args:
            start_date:     The date range(start).
            end_date:       The date range(end).
            ticker_list:    `list` of The target tickers.
        '''
        raise NotImplementedError()
