# -*- coding: utf-8 -*-
from algowars.extractable_historical_data import ExtractableHistoricalData
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import urllib.request
from time import sleep
import yfinance as yf
from logging import getLogger


class FacadeYFinance(ExtractableHistoricalData):
    '''
    Using Yahoo Finance API, 
    this class make it possible to load, save, and extract the historical data.
    '''

    # Logs dir path.
    __logs_dir = "logs/historical/"
    # CSV file name of ticker master data.
    __ticker_master_path = "masterdata/ticker_master_crypto.csv"
    # Format of date.
    __date_format = "%Y-%m-%d"

    def __init__(
        self,
        logs_dir=None,
        date_format="%Y-%m-%d",
        sleep_sec=60,
        ticker_master_path="masterdata/ticker_master_crypto.csv",
    ):
        '''
        Init.

        Args:
            logs_dir:               `str` of path to the directory 
                                    in which the historical data files will be saved.
            
            date_format:            `str` of date format.
            sleep_sec:              `int` of the number of seconds to sleep.
            ticker_master_path:     `str` of path to ticker master file.
        '''
        self.__logger = getLogger("algowars")
        if logs_dir is not None:
            self.__logs_dir = logs_dir
        self.__date_format = date_format

        if isinstance(sleep_sec, int) is False:
            raise TypeError("The type of `sleep_sec` must be `int`.")
        if sleep_sec <= 5:
            raise TypeError("The value of `sleep_sec` must be more than `5`.")

        self.__ticker_master_path = ticker_master_path
        self.__sleep_sec = sleep_sec
    
    def load(self, target_ticker=None):
        '''
        Load and save histroical data into local csv file.
        
        Args:
            target_ticker:      Target ticker. If `None`, the data of all ticker is loaded.
        '''
        if target_ticker is not None:
            self.__get_and_sleep([target_ticker])
        else:
            df = pd.read_csv(self.__ticker_master_path)
            ticker_list = df.ticker.values.tolist()
            self.__get_and_sleep(ticker_list)

    def __get_and_sleep(self, ticker_list):
        self.__logger.info("Ticker symbol: " + str(", ".join(ticker_list)))

        df = yf.download(ticker_list)
        idx = pd.IndexSlice

        min_timestamp_list = []
        for ticker in ticker_list:
            ticker_df = df.loc[:,idx[:, ticker]]
            ticker_df = ticker_df.dropna()
            ticker_df.columns = ticker_df.columns.droplevel(1)
            ticker_df["timestamp"] = ticker_df.index
            ticker_df["adjusted_close"] = ticker_df["Adj Close"]
            ticker_df["close"] = ticker_df["Close"]
            ticker_df["high"] = ticker_df["High"]
            ticker_df["low"] = ticker_df["Low"]
            ticker_df["open"] = ticker_df["Open"]
            ticker_df["volume"] = ticker_df["Volume"]
            ticker_df["dividend_amount"] = 0.0
            ticker_df["split_coefficient"] = 0.0
            
            ticker_df = ticker_df[
                [
                    "timestamp",
                    "open", 
                    "high", 
                    "low", 
                    "close", 
                    "adjusted_close", 
                    "volume", 
                    "dividend_amount", 
                    "split_coefficient"
                ]
            ]
            ticker_df.index = pd.to_datetime(ticker_df["timestamp"], format="%Y-%m-%d")
            ticker_df = ticker_df.resample('D').mean().fillna(method='ffill')
            ticker_df["timestamp"] = ticker_df.index
            ticker_df = ticker_df[
                [
                    "timestamp",
                    "open", 
                    "high", 
                    "low", 
                    "close", 
                    "adjusted_close", 
                    "volume", 
                    "dividend_amount", 
                    "split_coefficient"
                ]
            ]
            min_timestamp_list.append(
                ticker_df.timestamp.min()
            )
            ticker_df.to_csv(self.__logs_dir + ticker + ".csv", index=False)
            self.__logger.info("The downloading " + ticker + " is end.")

        ticker_df = pd.DataFrame(ticker_list)
        ticker_df.columns = ["ticker"]
        min_timestamp_df = pd.DataFrame(min_timestamp_list)
        min_timestamp_df.columns = ["min_timestamp"]

        self.__logger.debug(pd.concat([ticker_df, min_timestamp_df], axis=1).sort_values(by=["min_timestamp"]))

    def extract(self, start_date, end_date, ticker_list):
        '''
        Extract histroical data.

        Args:
            start_date:     The date range(start).
            end_date:       The date range(end).
            ticker_list:    `list` of The target tickers.
        
        Returns:
            `pd.DataFrame`.
        '''
        df_list = [None]
        for i in range(len(ticker_list)):
            df = pd.read_csv(self.__logs_dir + ticker_list[i] + ".csv")
            df["ticker"] = ticker_list[i]
            df_list.append(df)

        result_df = pd.concat(df_list)

        result_df = result_df[[
            "adjusted_close",
            "close",
            "high",
            "low",
            "open",
            "volume",
            "timestamp",
            "ticker"
        ]]

        result_df = result_df.dropna()

        result_df["date"] = result_df["timestamp"]
        result_df["timestamp"] = result_df.date.apply(self.__get_timestamp)

        if start_date is not None:
            start_timestamp = datetime.strptime(start_date, self.__date_format).timestamp()
            result_df = result_df[result_df.timestamp >= start_timestamp]
        if end_date is not None:
            end_timestamp = datetime.strptime(end_date, self.__date_format).timestamp()
            result_df = result_df[result_df.timestamp <= end_timestamp]

        date_df = result_df.sort_values(by=["timestamp"]).drop_duplicates(["date"])
        date_df = date_df.reset_index()
        date_df = pd.concat([
            date_df,
            pd.DataFrame(np.arange(date_df.shape[0]), columns=["date_key"])
        ], axis=1)
        
        result_df = pd.merge(
            result_df,
            date_df[["date", "date_key"]],
            on="date"
        )

        result_df = result_df.sort_values(by=["timestamp", "ticker"])

        return result_df

    def __get_timestamp(self, v):
        return datetime.strptime(v, self.__date_format).timestamp()

def Main(params_dict):
    '''
    Entry points.
    
    Args:
        params_dict:    `dict` of parameters.
    '''
    target_ticker = params_dict["target_ticker"]
    del params_dict["target_ticker"]

    # not logging but logger.
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
    logger = getLogger("algowars")
    handler = StreamHandler()
    if params_dict["verbose"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)
    logger.addHandler(handler)

    del params_dict["verbose"]
    FacadeYFinance(**params_dict).load(target_ticker)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load data from Yahoo Finance API.")
    
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="Verbose mode or not."
    )

    parser.add_argument(
        "-ld",
        "--logs_dir",
        type=str,
        default="logs/historical/",
        help="Logs dir."
    )

    parser.add_argument(
        "-tt",
        "--target_ticker",
        type=str,
        default=None,
        help="Target ticker symbol."
    )

    parser.add_argument(
        "-tmp",
        "--ticker_master_path",
        type=str,
        default="masterdata/ticker_master_crypto.csv",
        help="Path to the ticker master data."
    )

    args = parser.parse_args()
    params_dict = vars(args)
    Main(params_dict)
