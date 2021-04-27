# -*- coding: utf-8 -*-
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from datetime import datetime
from accelbrainbase.samplabledata.noise_sampler import NoiseSampler
from algowars.extractable_historical_data import ExtractableHistoricalData


class VolatilityConditionalNoiseSampler(NoiseSampler):
    '''
    Sampler which draws samples from the `fake` distribution of volatility.
    '''

    # Target features
    __target_features_list = [
        "adjusted_close",
        "close",
        "high",
        "low",
        "open",
        "volume"
    ]

    # Format of date.
    __date_format = "%Y-%m-%d"

    # Start date.
    __start_date = None

    # End date.
    __end_date = None

    # Fix date or not.
    __fix_date_flag = False

    def __init__(
        self, 
        extractable_historical_data,
        ticker_list,
        start_date,
        end_date,
        batch_size=20, 
        seq_len=10, 
        channel=3,
        target_features_list=None,
        diff_mode=False,
        log_mode=False,
        lstm_mode=True,
        ctx=mx.gpu()
    ):
        '''
        Init.

        Args:
            extractable_historical_data:    is-a `ExtractableHistoricalData`.
            ticker_list:                    `list` of tickers.
            start_date:                     `str` of start date.
            end_date:                       `str` of end date.
            batch_size:                     Batch size.
            seq_len:                        The length of sequneces.
            channel:                        Channel.
            target_features_list:            `list` of target feature list.
            diff_mode:                      `bool`. If `True`, this class outputs difference sequences.
            log_mode:                       `bool`. If `True`, this class outputs logarithmic rates of change.
            lstm_mode:                      `bool`. If `True`, this class converts data for LSTM model.
            ctx:                            `mx.gpu()` or `mx.cpu()`. 

        '''
        if isinstance(extractable_historical_data, ExtractableHistoricalData) is False:
            raise TypeError()

        self.__extractable_historical_data = extractable_historical_data

        self.__batch_size = batch_size
        self.__seq_len = seq_len
        self.__channel = channel
        self.__ticker_list = ticker_list
        self.__start_date = start_date
        self.__end_date = end_date
        if target_features_list is not None:
            self.__target_features_list = target_features_list
        self.__dim = len(self.__target_features_list)
        if lstm_mode is True and len(target_features_list) > 1:
            raise ValueError()
        self.__lstm_mode = lstm_mode

        self.__diff_mode = diff_mode
        self.__normlized_flag = False
        self.__log_mode = log_mode
        self.setup_data()
        self.__ctx = ctx

    def sigmoid(self, x):
        if self.__diff_mode is False:
            x = x / 100
        y = 1.0 / (1.0 + np.exp(-x))
        return y

    def logit(self, y):
        if self.__diff_mode is False:
            const = 100
        else:
            const = 1

        return (np.log(y / (1 - y)) * const)

    def setup_data(self):
        self.__stock_df = self.__extractable_historical_data.extract(
            self.__start_date,
            self.__end_date,
            self.__ticker_list
        )
        
        self.__date_df = self.__stock_df.date.drop_duplicates()
        self.__timestamp_list = self.__stock_df.timestamp.drop_duplicates().values.tolist()

        if self.__diff_mode is True:
            df_list = []
            for ticker in self.__ticker_list:
                df = self.__stock_df[self.__stock_df.ticker == ticker]
                if self.__log_mode is True:
                    for target_feature in self.__target_features_list:
                        df.loc[:, target_feature] = (
                            df[target_feature] / df[target_feature].shift(1).fillna(df[target_feature])
                        ).apply(np.log)
                else:
                    for target_feature in self.__target_features_list:
                        df.loc[:, target_feature] = (
                            df[target_feature] - df[target_feature].shift(1).fillna(df[target_feature])
                        )

                df_list.append(df)
            self.__stock_df = pd.concat(df_list)

        if self.__stock_df.shape[0] == 0:
            print((self.__start_date, self.__end_date))
            raise ValueError("In setup, stock df is not found.")

    def draw(self):
        '''
        Generate noise samples.
        
        Returns:
            `np.ndarray` of samples.
        '''
        pre_sampled_arr = np.zeros((self.__batch_size, self.__channel, self.__seq_len, self.__dim))

        for batch in range(self.__batch_size):
            for _i in range(10):
                try:
                    if self.__fix_date_flag is False:
                        row = np.random.randint(
                            low=0, 
                            high=self.__date_df.shape[0] - (self.__seq_len * 2)
                        )
                    else:
                        row = 0

                    pre_start_timestamp = self.__timestamp_list[row]
                    try:
                        pre_end_timestamp = self.__timestamp_list[row+self.__seq_len]
                    except IndexError:
                        continue

                    date_list = self.__date_df[row:row+self.__seq_len+10].astype(str).values.tolist()
                    pre_stock_df = self.__stock_df[
                        (self.__stock_df.timestamp >= pre_start_timestamp) & (self.__stock_df.timestamp < pre_end_timestamp)
                    ]

                    pre_sampled_arr = self.__t(pre_sampled_arr, batch, pre_stock_df, date_list)
                    break
                except:
                    if _i >= 9:
                        raise

                    continue

        pre_sampled_arr = nd.ndarray.array(pre_sampled_arr, ctx=self.__ctx)

        if self.__lstm_mode is True:
            pre_sampled_arr = pre_sampled_arr.reshape((
                pre_sampled_arr.shape[0],
                pre_sampled_arr.shape[2],
                pre_sampled_arr.shape[1],
            ))

        return pre_sampled_arr

    def __date_to_timestamp(self, date_str):
        start_date = date_str + " 00:00:00"
        end_date = date_str + " 23:59:59"
        start_timestamp = datetime.strptime(start_date, self.__date_format + " %H:%M:%S").timestamp()
        end_timestamp = datetime.strptime(end_date, self.__date_format + " %H:%M:%S").timestamp()
        return float(start_timestamp), float(end_timestamp)

    def __t(self, sample_arr, batch, stock_df, date_list):
        stock_df = stock_df.sort_values(by=["ticker"])
        i = 0

        for seq in range(len(date_list)):
            start_timestamp, end_timestamp = self.__date_to_timestamp(date_list[seq])

            seq_df = stock_df[
                (stock_df.timestamp >= start_timestamp) & (stock_df.timestamp <= end_timestamp)
            ][self.__target_features_list]
            if seq_df.shape[0] == sample_arr[batch, :].shape[0]:
                try:
                    try:
                        sample_arr[batch, :, i] = seq_df.values.reshape(sample_arr[batch, :, i].shape)
                    except ValueError:
                        sample_arr[batch, :, i] = seq_df.values
                    i += 1
                except ValueError:
                    print("seq_df.values")
                    print(seq_df.values[:10])
                    print(stock_df.ticker.drop_duplicates().values)
                    print(seq_df.ticker.drop_duplicates().values)
                    print(date_list[seq])
                    print((start_timestamp, end_timestamp))
                    raise
            else:
                if seq_df.shape[0] > 0:
                    print("The data was not found.")
                    print(stock_df[
                        (stock_df.timestamp >= start_timestamp) & (stock_df.timestamp <= end_timestamp)
                    ].ticker.drop_duplicates().values)
                    print(stock_df.ticker.drop_duplicates().values)
                    print(date_list[seq])
                    print((start_timestamp, end_timestamp))
                    print((
                        seq_df.shape[0],
                        sample_arr[batch, :].shape[0]
                    ))
        return sample_arr

    def get_start_date(self):
        ''' getter '''
        return self.__start_date
    
    def set_start_date(self, value):
        ''' setter '''
        self.__start_date = value
    
    start_date = property(get_start_date, set_start_date)

    def get_end_date(self):
        ''' getter '''
        return self.__end_date
    
    def set_end_date(self, value):
        ''' setter '''
        self.__end_date = value
    
    end_date = property(get_end_date, set_end_date)

    def get_fix_date_flag(self):
        return self.__fix_date_flag
    
    def set_fix_date_flag(self, value):
        self.__fix_date_flag = value
    
    fix_date_flag = property(get_fix_date_flag, set_fix_date_flag)

    def get_seq_len(self):
        return self.__seq_len
    
    def set_seq_len(self, value):
        self.__seq_len = value

    seq_len = property(get_seq_len, set_seq_len)
