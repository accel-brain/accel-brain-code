# -*- coding: utf-8 -*-
from algowars.extractable_historical_data import ExtractableHistoricalData
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
from time import sleep
from logging import getLogger


class FacadeAlphaVantage(ExtractableHistoricalData):
    '''
    Using AlphaVantage API, 
    this class make it possible to load, save, and extract the historical data.
    '''

    # Logs dir path.
    __logs_dir = "logs/historical/"
    # CSV file name of ticker master data.
    __ticker_master_path = "masterdata/ticker_master.csv"
    # Format of date.
    __date_format = "%Y-%m-%d"
    # URL
    __url_format = "https://www.alphavantage.co/query?function={@function@}&symbol={@symbol@}&apikey={@apikey@}&datatype={@datatype@}&indexing_type={@indexing_type@}&outputsize={@outputsize@}"

    def get_url_format(self):
        return self.__url_format
    
    def set_url_format(self, value):
        self.__url_format = value

    url_format = property(get_url_format, set_url_format)

    def __init__(
        self,
        api_key,
        logs_dir=None,
        outputsize="compact",
        date_format="%Y-%m-%d",
        av_function="TIME_SERIES_DAILY_ADJUSTED",
        sleep_sec=10,
        ticker_master_path="masterdata/ticker_master.csv",
        crypto_mode=False
    ):
        '''
        Init.

        Args:
            api_key:                `str` of api key.
            logs_dir:               `str` of path to the directory 
                                    in which the historical data files will be saved.

            outputsize:             `str` of output size. (Default: compact).
                                    If you want to get all data, input `full`.
            
            date_format:            `str` of date format.
            av_function:            `str`. `TIME_SERIES_DAILY_ADJUSTED` or `TIME_SERIES_DAILY`.
            sleep_sec:              `int` of the number of seconds to sleep.
            ticker_master_path:     `str` of path to ticker master file.
        '''
        self.__logger = getLogger("algowars")
        if logs_dir is not None:
            self.__logs_dir = logs_dir
        self.__date_format = date_format
        self.__outputsize = outputsize
        self.__av_function = av_function
        self.__indexing_type = "date"

        if isinstance(sleep_sec, int) is False:
            raise TypeError("The type of `sleep_sec` must be `int`.")
        if sleep_sec <= 5:
            raise TypeError("The value of `sleep_sec` must be more than `5`.")

        self.__ticker_master_path = ticker_master_path
        self.__sleep_sec = sleep_sec
        self.__api_key = api_key
        self.__crypto_mode = crypto_mode
    
    def load(self, target_ticker=None):
        '''
        Load and save histroical data into local csv file.
        
        Args:
            target_ticker:      Target ticker. If `None`, the data of all ticker is loaded.
        '''
        if target_ticker is not None:
            self.__get_and_sleep(target_ticker)
        else:
            df = pd.read_csv(self.__ticker_master_path)
            for ticker in df.ticker.values:
                if target_ticker is not None and ticker != target_ticker:
                    continue
                self.__get_and_sleep(ticker)

    def __get_and_sleep(self, ticker):
        self.__logger.info("Sleeeeeeeeeeeeping ... (" + str(self.__sleep_sec) + " sec)")

        sleep(self.__sleep_sec)

        params_dict = {
            "function": self.__av_function,
            "symbol": ticker,
            "apikey": self.__api_key,
            "datatype": "csv",
            "indexing_type": self.__indexing_type,
            "outputsize": self.__outputsize
        }
        url = self.__url_format
        for k, v in params_dict.items():
            url = url.replace("{@" + k + "@}", v)

        self.__logger.info("Ticker symbol: " + str(ticker))
        self.__logger.info("Target URL: " + str(url))

        #req = urllib.request.Request(url=url, data=data)
        req = urllib.request.Request(url=url)
        res = urllib.request.urlopen(req)
        with open(self.__logs_dir + ticker + ".csv", "w", newline=None) as f:
            f.write(res.read().decode("utf-8"))
            f.close()
            self.__logger.info("End.")

        if self.__crypto_mode is True:
            df = pd.read_csv(self.__logs_dir + ticker + ".csv")
            df = df[[
                "timestamp",
                "open (USD)",
                "high (USD)",
                "low (USD)",
                "close (USD)",
                "volume",
                "market cap (USD)"
            ]]
            df["adjusted_close"] = df["close (USD)"]
            df = df[[
                "timestamp",
                "open (USD)",
                "high (USD)",
                "low (USD)",
                "close (USD)",
                "adjusted_close",
                "volume",
            ]]
            df["dividend_amount"] = -1
            df["split_coefficient"] = -1
            df.columns = [
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
            df.to_csv(self.__logs_dir + ticker + ".csv", index=False)

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
        #self.__logger.debug("total: " + str(result_df.shape[0]))

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
        #self.__logger.debug("After dropping na: " + str(result_df.shape[0]))

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
    with open(params_dict["api_key"], "r") as f:
        params_dict["api_key"] = f.read()
        f.close()
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

    api = FacadeAlphaVantage(**params_dict)
    if params_dict["crypto_mode"] is True:
        # Crypto URL
        crypto_url_format = "https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={@symbol@}&apikey={@apikey@}&market=CNY&datatype=csv"
        api.url_format = crypto_url_format

    api.load(target_ticker)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load data from AlphaVantage API.")
    
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        default=False,
        help="Verbose mode or not."
    )

    parser.add_argument(
        "-a",
        "--api_key",
        type=str,
        default="config/alpha_vantage_api_key.txt",
        help="Path of file that contains the api key."
    )

    parser.add_argument(
        "-os",
        "--outputsize",
        type=str,
        default="full",
        help="Output size. (Default: full) If you want to get all data, `full`. "
    )

    parser.add_argument(
        "-ld",
        "--logs_dir",
        type=str,
        default="logs/historical/",
        help="Logs dir."
    )

    parser.add_argument(
        "-df",
        "--date_format",
        type=str,
        default="%Y-%m-%d",
        help="Format of date."
    )

    parser.add_argument(
        "-tt",
        "--target_ticker",
        type=str,
        default=None,
        help="Target ticker symbol."
    )

    parser.add_argument(
        "-cm",
        "--crypto_mode",
        action="store_true",
        default=False,
        help="Crypto or not."
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
