# -*- coding: utf-8 -*-
from algowars.extractable_historical_data import ExtractableHistoricalData
from algowars.volatility_gan import VolatilityGAN
import pandas as pd
import numpy as np
from datetime import datetime
import urllib.request
from time import sleep


class ExtractableGeneratedHistoricalData(ExtractableHistoricalData):
    '''
    The extractor of generated historical data.
    '''

    __logs_dir = "logs/"

    def get_logs_dir(self):
        return self.__logs_dir
    
    def set_logs_dir(self, value):
        self.__logs_dir = value

    logs_dir = property(get_logs_dir, set_logs_dir)

    __master_dir = "masterdata/"

    def get_master_dir(self):
        ''' getter '''
        return self.__master_dir
    
    def set_master_dir(self, value):
        ''' setter '''
        self.__master_dir = value

    master_dir = property(get_master_dir, set_master_dir)

    __original_start_date = None

    def get_original_start_date(self):
        ''' getter '''
        return self.__original_start_date
    
    def set_original_start_date(self, value):
        ''' setter '''
        self.__original_start_date = value

    original_start_date = property(get_original_start_date, set_original_start_date)

    __original_end_date = None

    def get_original_end_date(self):
        ''' getter '''
        return self.__original_end_date

    def set_original_end_date(self, value):
        ''' setter '''
        self.__original_end_date = value

    original_end_date = property(get_original_end_date, set_original_end_date)

    __not_learning_flag = False

    def get_not_learning_flag(self):
        ''' getter '''
        return self.__not_learning_flag
    
    def set_not_learning_flag(self, value):
        ''' setter '''
        self.__not_learning_flag = value

    not_learning_flag = property(get_not_learning_flag, set_not_learning_flag)

    def __init__(
        self, 
        extractable_historical_data, 
        batch_size,
        seq_len,
        learning_rate,
        item_n,
        k_step,
        original_start_date,
        original_end_date,
        logs_dir=None,
        g_params_path="params/recursive_seq_2_seq_model_re_encoder_model.params",
        re_e_params_path="params/recursive_seq_2_seq_model_model.params",
        d_params_path="params/discriminative_model_model.params",
        transfer_flag=True,
        not_learning_flag=False,
        diff_mode=True,
        log_mode=True,
        multi_fractal_mode=True,
        long_term_seq_len=30
    ):
        '''
        Init.

        Args:
            extractable_historical_data:    is-a `ExtractableHistoricalData`.
            batch_size:                     `int` of batch size.
            seq_len:                        `int` of the number of sequence.
            learning_rate:                  `float` of learning rate.
            item_n:                         `int` of the number of generator's learning iteration in the volatility GAN.
            k_step:                         `int` of the number of discriminator's learning iteration in the volatility GAN.
            original_start_date:            `str` of start date of generated historical data.
            original_end_date:              `str` of end date of generated historical data.
            ticker_list:                    `list` of tickers.
            logs_dir:                       `str` of path to directory in which the historical data files were saved.
            g_params_path:                  `str` of path to generator's learned parameters.
            re_e_params_path:               `str` of path to re-encoder's learned parameters.
            d_params_path:                  `str` of path to discriminator's learned parameters.
            transfer_flag:                  `bool`. If `True`, this class will do transfer learning.
            not_learning_flag:              `bool`. If `True`, the learning of volatility GAN will not be done.
            multi_fractal_mode:             `bool`.
            long_term_seq_len:              `int`.
        '''
        if isinstance(extractable_historical_data, ExtractableHistoricalData) is False:
            raise TypeError()
        self.__extractable_historical_data = extractable_historical_data

        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__learning_rate = learning_rate
        self.__item_n = item_n
        self.__k_step = k_step
        self.__original_start_date = original_start_date
        self.__original_end_date = original_end_date

        self.__g_params_path = g_params_path
        self.__re_e_params_path = re_e_params_path
        self.__d_params_path = d_params_path
        self.__transfer_flag = transfer_flag

        if logs_dir is not None:
            self.__logs_dir = logs_dir

        self.__not_learning_flag = not_learning_flag
        self.__diff_mode = diff_mode
        self.__log_mode = log_mode
        self.__multi_fractal_mode = multi_fractal_mode
        self.__long_term_seq_len = long_term_seq_len

    def extract(self, start_date, end_date, ticker_list):
        '''
        Extract histroical data from local csv file.

        Args:
            start_date:                 The date range(start) in original data.
            end_date:                   The date range(end) in original darta.
            ticker_list:                `list` of The target tickers.

        Returns:
            `pd.DataFrame`.
        '''
        df = self.__extractable_historical_data.extract(
            start_date, 
            end_date, 
            ticker_list
        )
        self.__pre_updated_df = df
        df = self.__update_from_generated(
            df,
            start_date, 
            end_date, 
            ticker_list
        )
        df = df.drop_duplicates(["ticker", "batch_n", "date"])
        return df

    def __update_from_generated(self, df, start_date, end_date, ticker_list):
        '''
        Update data from generated.

        Args:
            start_date:     The date range(start).
            end_date:       The date range(end).
            ticker_list:    `list` of The target tickers.
        
        Returns:
            Updated `pd.DataFrame`.
        '''
        self.__volatility_GAN = VolatilityGAN(
            extractable_historical_data=self.__extractable_historical_data,
            ticker_list=ticker_list,
            start_date=self.__original_start_date,
            end_date=self.__original_end_date,
            batch_size=self.__batch_size,
            seq_len=self.__seq_len,
            learning_rate=self.__learning_rate,
            g_params_path=self.__g_params_path,
            re_e_params_path=self.__re_e_params_path,
            d_params_path=self.__d_params_path,
            transfer_flag=self.__transfer_flag,
            diff_mode=self.__diff_mode,
            log_mode=self.__log_mode,
            multi_fractal_mode=self.__multi_fractal_mode,
            long_term_seq_len=self.__long_term_seq_len
        )
        if self.__not_learning_flag is False:
            self.__volatility_GAN.learn(iter_n=self.__item_n, k_step=self.__k_step)

            self.__volatility_GAN.save_parameters(
                g_params_path=self.__g_params_path,
                re_e_params_path=self.__re_e_params_path,
                d_params_path=self.__d_params_path
            )

            d_logs_list, g_logs_list = self.__volatility_GAN.extract_logs()
            feature_matching_arr = self.__volatility_GAN.extract_feature_matching_logs()

            np.save(self.__logs_dir + "d_logs", d_logs_list)
            np.save(self.__logs_dir + "g_logs", g_logs_list)
            np.save(self.__logs_dir + "feature_matching_logs", feature_matching_arr)

        generated_stock_df_list, true_stock_df, rest_generated_stock_df_list = self.__volatility_GAN.inference(
            start_date,
            end_date,
            ticker_list,
        )
        pd.concat(rest_generated_stock_df_list).to_csv(self.__logs_dir + "rest_generated_stock_df.csv", index=False)

        df_list = []
        for i in range(len(generated_stock_df_list)):
            for ticker in ticker_list:
                _generated_stock_df = generated_stock_df_list[i][
                    generated_stock_df_list[i].ticker == ticker
                ]

                _df = df[df.ticker == ticker]
                _df["batch_n"] = str(i)

                adjusted_close_df = _generated_stock_df["adjusted_close"].iloc[:_df.shape[0]]

                col_list = [col for col in _df.columns if col != "adjusted_close"]
                _df = pd.concat([
                    _df[col_list].reset_index(),
                    adjusted_close_df.reset_index()
                ], axis=1)

                df_list.append(_df)

        df = pd.concat(df_list)
        return df


def Main(params_dict):
    '''
    Entry points.
    
    Args:
        params_dict:    `dict` of parameters.
    '''

    from algowars.extractablehistoricaldata.facade_alpha_vantage import FacadeAlphaVantage
    from algowars.extractablehistoricaldata.extractable_generated_historical_data import ExtractableGeneratedHistoricalData
    import pandas as pd
    # not logging but logger.
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

    logger = getLogger("accelbrainbase")
    handler = StreamHandler()
    if params_dict["verbose"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)
    logger.addHandler(handler)

    logger = getLogger("algowars")
    handler = StreamHandler()
    if params_dict["verbose"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)
    logger.addHandler(handler)

    with open(params_dict["api_key"], "r") as f:
        params_dict["api_key"] = f.read()
        f.close()

    if params_dict["verbose"] is True:
        print("Load historical data.")

    extractable_historical_data = FacadeAlphaVantage(
        api_key=params_dict["api_key"],
        logs_dir=self.__logs_dir,
    )

    extractable_generated_historical_data = ExtractableGeneratedHistoricalData(
        extractable_historical_data=extractable_historical_data,
        batch_size=params_dict["batch_size"],
        seq_len=params_dict["seq_len"],
        learning_rate=params_dict["learning_rate"],
        item_n=params_dict["item_n"],
        k_step=params_dict["k_step"],
        original_start_date=params_dict["start_date"],
        original_end_date=params_dict["end_date"],
    )

    stock_master_df = pd.read_csv(params_dict["ticker_master_path"])
    ticker_list = stock_master_df.ticker.values.tolist()

    generated_historical_df = extractable_generated_historical_data.extract(
        start_date=params_dict["generated_start_date"],
        end_date=params_dict["generated_end_date"],
        ticker_list=ticker_list
    )
    generated_historical_df.to_csv(self.__logs_dir + "/generated_historical_df.csv", index=False)
    print("end")

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
        "-ld",
        "--logs_dir",
        type=str,
        default="logs/",
        help="Logs dir."
    )

    parser.add_argument(
        "-tmp",
        "--ticker_master_path",
        type=str,
        default="masterdata/ticker_master.csv",
        help="Path to ticker master data."
    )

    parser.add_argument(
        "-sd",
        "--start_date",
        type=str,
        default=None,
        help="Start date."
    )

    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        default=None,
        help="End date."
    )

    parser.add_argument(
        "-sc",
        "--stock_choiced",
        type=str,
        default="all",
        help="`all`, `random`, or `x,y,z`."
    )

    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=20,
        help="batch size."
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        type=int,
        default=7,
        help="The length of sequences."
    )
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        type=int,
        default=100,
        help="The number of units in hidden layers."
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-05,
        help="Learnign rate."
    )

    parser.add_argument(
        "-in",
        "--item_n",
        type=int,
        default=1000,
        help="The number of training D."
    )
    parser.add_argument(
        "-ks",
        "--k_step",
        type=int,
        default=10,
        help="The number of training G."
    )
    parser.add_argument(
        "-gsd",
        "--generated_start_date",
        type=str,
        default=None,
        help="Start date generated."
    )
    parser.add_argument(
        "-ged",
        "--generated_end_date",
        type=str,
        default=None,
        help="End date generated."
    )

    args = parser.parse_args()
    params_dict = vars(args)
    Main(params_dict)
