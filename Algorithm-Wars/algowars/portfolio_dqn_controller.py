# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib

from algowars.exception.stock_history_error import StockHistoryError

from algowars.extractable_historical_data import ExtractableHistoricalData

from algowars.noisesampler.volatility_conditional_noise_sampler import VolatilityConditionalNoiseSampler
from algowars.generativemodel.recursive_seq2seq_model import RecursiveSeq2SeqModel
from algowars.controllablemodel.gancontroller.volatility_gan_controller import VolatilityGANController

from algowars.policysampler.portfolio_percent_policy import PortfolioPercentPolicy

from algowars.technicalobserver.multi_trade_observer import MultiTradeObserver
from algowars.technicalobserver.bollinger_band_observer import BollingerBandObserver
from algowars.technicalobserver.contrarian_bollinger_band_observer import ContrarianBollingerBandObserver
from algowars.technicalobserver.contrarian_rsi_observer import ContrarianRSIObserver
from algowars.technicalobserver.rsi_observer import RSIObserver
from algowars.technicalobserver.macd_observer import MACDObserver

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss

from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks

from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.lstm_networks import LSTMNetworks
from accelbrainbase.observabledata._mxnet.lstmnetworks.encoder_decoder import EncoderDecoder

from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generativemodel.encoder_decoder import EncoderDecoder as GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.truesampler.normal_true_sampler import NormalTrueSampler

from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler

from accelbrainbase.controllablemodel._mxnet.gancontroller.aae_controller import AAEController

from accelbrainbase.controllablemodel._mxnet.dqlcontroller.dqn_controller import DQNController
from accelbrainbase.observabledata._mxnet.functionapproximator.function_approximator import FunctionApproximator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.mobilenet_v2 import MobileNetV2

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm

from algowars.extractablehistoricaldata.facade_alpha_vantage import FacadeAlphaVantage

import pandas as pd
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from logging import getLogger


class PortfolioDQNController(object):
    '''
    The controller of the Portfolio DQN.
    '''
    
    __ticker_list = []

    def get_ticker_list(self):
        return self.__ticker_list
    
    def set_ticker_list(self, value):
        self.__ticker_list = value

    ticker_list = property(get_ticker_list, set_ticker_list)

    def __init__(
        self,
        extractable_historical_data,
        stock_master_df,
        date_fraction,
        start_date,
        end_date,
        generated_start_date,
        generated_end_date,
        learning_rate,
        defualt_money,
        batch_size=20,
        channel=3,
        seq_len=10,
        rebalance_prob=0.3,
        rebalancing_prob_mean=0.3,
        rebalancing_prob_std=0.01,
        risk_free_rate_mean=0.3,
        risk_free_rate_std=0.01,
        next_action_n=10,
        first_date_key=0,
        agent_n=10,
        epsilon_greedy_rate=0.7,
        alpha_value=1.0,
        gamma_value=0.1,
        technical_flag=False,
        logs_dir="",
        result_dir="result/",
        extractable_generated_historical_data=None,
        transfer_flag=False,
        target_features_list=["adjusted_close"],
        hybridize_flag=True,
        ctx=mx.gpu(),
    ):
        '''
        Init.

        Args:
            extractable_historical_data:    is-a `ExtractableHistoricalData`.
            stock_master_df:                `pd.DataFrame` of stock master data.
            date_fraction:                  `int` of date fraction.
            start_date:                     `str` of start date of learning data.
            end_date:                       `str` of end date of learning data.
            generated_start_date:           `str` of start date of generated data.
            generated_end_date:             `str` of end date of generated data.
            learning_rate:                  `float` of learning rate.
            defualt_money:                  `flaot` of the first average money each agent owns.
            batch_size:                     `int` of batch size.
            channel:                        `int` of image channel.
            seq_len:                        `int` of the length of sequence.
            rebalance_prob:                 `float` of the probability of each agent rebalancing.
            rebalancing_prob_mean:          `float` of the average probability of each agent rebalancing.
            rebalancing_prob_std:           `float` of the standard deviation of the probability of each agent rebalancing.
            risk_free_rate_mean:            `float` of the average risk free rate.
            risk_free_rate_std:             `float` of the standard deviation of the risk free rate.
            next_action_n:                  `int` of the number of agent's action choices.
            first_date_key:                 `int` of the first date key.
            agent_n:                        `int` of the number of agent.
            epsilon_greedy_rate:            `float` of the epsilon greedy rate.
            alpha_value:                    `float` of alpha value in DQN.
            gamma_value:                    `float` of gamma value in DQN.
            logs_dir:                       `str` of path to directory in which the historical data files were saved.
            extractable_generated_historical_data:  is-a `ExtractableGeneratedHistoricalData`.
            transfer_flag:                  `bool`. If `True`, this class will do transfer learning.
            target_features_list:           `list` of `str`. The value is ...
                                                - `adjusted_close`: adjusted close.
                                                - `close`: close.
                                                - `high`: high value.
                                                - `low`: low value.
                                                - `open`: open.
                                                - `volume`: volume.
            hybridize_flag:                 Call `mxnet.gluon.HybridBlock.hybridize()` or not. 
            ctx:                            `mx.gpu()` or `mx.cpu()`.
        '''
        logger = getLogger("algowars")
        self.__logger = logger

        self.__logger.debug("date_fraction: " + str(date_fraction))
        self.__logger.debug("seq_len: " + str(seq_len))

        diff_mode = True
        tanh_mode = False
        z_score_mode = False
        min_max_mode = False
        log_mode = True
        logistic_mode = False

        initializer = mx.initializer.Uniform(1.0)

        if isinstance(extractable_historical_data, ExtractableHistoricalData) is False:
            raise TypeError()

        self.__extractable_historical_data = extractable_historical_data
        self.__extractable_generated_historical_data = extractable_generated_historical_data
        self.__stock_master_df = stock_master_df
        self.__date_fraction = date_fraction
        self.__start_date = start_date
        self.__generated_start_date = generated_start_date
        self.__generated_end_date = generated_end_date
        self.__technical_flag = technical_flag

        ticker_list = stock_master_df.ticker.drop_duplicates().sort_values().values.tolist()

        self.__ticker_list = ticker_list

        self.__historical_df = self.__extractable_historical_data.extract(
            start_date=start_date,
            end_date=end_date,
            ticker_list=ticker_list
        )
        min_timestamp = 0
        min_timestamp_list = []
        for ticker in ticker_list:
            min_timestamp_list.append(
                self.__historical_df[self.__historical_df.ticker == ticker].timestamp.min()
            )
            min_timestamp = max([
                min_timestamp, 
                self.__historical_df[self.__historical_df.ticker == ticker].timestamp.min()
            ])

        start_timestamp = datetime.timestamp(
            datetime.strptime(
                start_date,
                "%Y-%m-%d"
            )
        )

        if start_timestamp < min_timestamp:
            print(ticker_list[min_timestamp_list.index(min_timestamp)])
            new_start_date = datetime.strftime(
                datetime.fromtimestamp(
                    min_timestamp,
                ), 
                "%Y-%m-%d"
            )
            raise ValueError("The start date must be changed to the minimum date of historical data with all targets (" + str(self.__start_date) + " -> " + str(new_start_date) + ")")

        test_historical_df = self.__extractable_historical_data.extract(
            start_date=self.__generated_start_date,
            end_date=self.__generated_end_date,
            ticker_list=ticker_list
        )
        test_historical_df = test_historical_df.sort_values(by=["date", "ticker"])

        if self.__extractable_generated_historical_data is not None:
            self.__generated_historical_df = self.__extractable_generated_historical_data.extract(
                start_date=start_date,
                end_date=end_date,
                ticker_list=self.__ticker_list
            )
            self.__generated_historical_df = self.__generated_historical_df.sort_values(by=["batch_n", "date"])
        else:
            self.__generated_historical_df = None

        if self.__extractable_generated_historical_data is not None:
            test_generated_historical_df = self.__extractable_generated_historical_data.extract(
                start_date=generated_start_date,
                end_date=generated_end_date,
                ticker_list=ticker_list
            )
            test_generated_historical_df = test_generated_historical_df.sort_values(by=["batch_n", "date"])
        else:
            test_generated_historical_df = None

        self.__test_historical_df = test_historical_df
        self.__test_generated_historical_df = test_generated_historical_df

        self.__historical_df = self.__historical_df.sort_values(by=["date", "ticker"])
        self.__generated_historical_df = self.__generated_historical_df.sort_values(by=["date", "ticker"])

        def _min_max(_df, _target_df, ticker_list):
            result_df_list = []
            for ticker in ticker_list:
                df = _df[_df.ticker == ticker]
                target_df = _target_df[_target_df.ticker == ticker]
                for col in ["adjusted_close", "close", "high", "low", "open"]:
                    df[col] = df[col].fillna(method='ffill')
                    df[col] = df[col].fillna(method='bfill')

                    _max, _min = target_df[col].max(), target_df[col].min()
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                    df[col] = (_max - _min) * df[col]
                    df[col] = df[col] + _min

                result_df_list.append(df)

            result_df = pd.concat(result_df_list)
            result_df = result_df.sort_values(by=["date", "ticker"])
            return result_df

        self.__generated_historical_df = _min_max(
            self.__generated_historical_df,
            self.__historical_df,
            ticker_list
        )

        self.__test_generated_historical_df = _min_max(
            self.__test_generated_historical_df,
            self.__test_historical_df,
            ticker_list
        )

        key = "_".join(ticker_list)
        md5_hash = hashlib.md5(key.encode()).hexdigest()

        self.__historical_df.to_csv(result_dir + "historical_data_" + str(md5_hash) + ".csv", index=False)
        self.__generated_historical_df.to_csv(result_dir + "generated_historical_data_" + str(md5_hash) + ".csv", index=False)
        self.__test_historical_df.to_csv(result_dir + "test_historical_data_" + str(md5_hash) + ".csv", index=False)
        self.__test_generated_historical_df.to_csv(result_dir + "test_generated_historical_data_" + str(md5_hash) + ".csv", index=False)

        generated_historical_df_list = [None] * batch_size
        for batch in range(batch_size):
            generated_historical_df_list[batch] = self.__generated_historical_df[
                self.__generated_historical_df.batch_n == str(batch)
            ]
        self.__generated_historical_df_list = generated_historical_df_list

        agents_master_arr, rebalance_policy_list, rebalance_sub_policy_list, timing_policy_list, money_arr = self.sample_multi_agents(
            batch_size,
            start_date,
            end_date,
            date_fraction,
            agent_n,
            defualt_money,
            rebalancing_prob_mean=rebalancing_prob_mean,
            rebalancing_prob_std=rebalancing_prob_std,
            risk_free_rate_mean=risk_free_rate_mean,
            risk_free_rate_std=risk_free_rate_std,
        )

        try:
            stock_master_df.ticker = stock_master_df.ticker.astype(int).astype(str)
        except:
            stock_master_df.ticker = stock_master_df.ticker.astype(str)

        stock_master_df = pd.merge(
            left=self.__stock_master_df,
            right=self.__historical_df[self.__historical_df.date_key == 0][["ticker", "adjusted_close"]],
            on="ticker"
        )
        stock_master_df = stock_master_df.sort_values(by=["ticker"])
        try:
            stock_master_df["date"] = stock_master_df["date"]
            _stock_master_df = stock_master_df[stock_master_df.date_key == 0]
            stock_master_arr = _stock_master_df[[
                "adjusted_close", 
                "commission", 
                "tax", 
                "ticker", 
                "expense_ratio", 
                "asset_allocation",
                "area_allocation",
                "yield",
                "date",
                "asset", 
                "area"
            ]].values
            daily_stock_master_flag = True
        except:
            stock_master_arr = stock_master_df[[
                "adjusted_close", 
                "commission", 
                "tax", 
                "ticker", 
                "expense_ratio", 
                "asset_allocation",
                "area_allocation",
                "yield",
                "asset", 
                "area"
            ]].values
            daily_stock_master_flag = False
            print("daily_stock_master_flag is False.")

        historical_arr = self.__historical_df[[
            "date_key",
            "adjusted_close",
            "volume",
            "ticker",
        ]].values

        for i in range(len(self.__generated_historical_df_list)):
            self.__generated_historical_df_list[i] = self.__generated_historical_df_list[i][
                [
                    "date_key",
                    "adjusted_close",
                    "volume",
                    "ticker"
                ]
            ]

        self.__logger.info(
            "rebalance policy data:"
        )
        self.__logger.info(
            rebalance_policy_list[0]
        )
        self.__logger.info(
            "rebalance sub-policy data:"
        )
        self.__logger.info(
            rebalance_sub_policy_list[0]
        )

        policy_sampler = PortfolioPercentPolicy(
            batch_size=batch_size,
            agents_master_arr=agents_master_arr,
            rebalance_policy_list=rebalance_policy_list,
            rebalance_sub_policy_list=rebalance_sub_policy_list,
            timing_policy_list=timing_policy_list,
            technical_observer_dict={
                "multi_trade_observer": MultiTradeObserver(
                    technical_observer_list=[
                        BollingerBandObserver(
                            time_window=20
                        ),
                        ContrarianBollingerBandObserver(
                            time_window=20
                        ),
                        ContrarianRSIObserver(
                            time_window=14,
                        ),
                        RSIObserver(
                            time_window=14,
                        ),
                        MACDObserver(
                            time_window=26*2,
                            long_term=26,
                            short_term=12,
                            signal_term=9,
                        ),
                    ]
                ),
                "bollinger_band_observer": BollingerBandObserver(
                    time_window=20
                ),
                "contrarian_bollinger_band_observer": ContrarianBollingerBandObserver(
                    time_window=20
                ),
                "contrarian_rsi_observer": ContrarianRSIObserver(
                    time_window=14,
                ),
                "rsi_observer": RSIObserver(
                    time_window=14,
                ),
                "macd_observer": MACDObserver(
                    time_window=26*2,
                    long_term=26,
                    short_term=12,
                    signal_term=9,
                )
            },
            historical_arr=historical_arr,
            stock_master_arr=stock_master_arr,
            rebalance_prob=rebalance_prob,
            next_action_n=next_action_n,
            first_date_key=first_date_key,
            date_fraction=date_fraction,
            generated_stock_df_list=self.__generated_historical_df_list,
            extractable_generated_historical_data=self.__extractable_generated_historical_data,
            ctx=ctx,
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            daily_stock_master_flag=daily_stock_master_flag,
            stock_master_df=stock_master_df,
        )

        dqn_initializer = None
        dqn_optimizer_name = "nadam"
        deep_q_learning = self.build_dqn(
            policy_sampler,
            learning_rate,
            epsilon_greedy_rate,
            alpha_value,
            gamma_value,
            dqn_initializer,
            dqn_optimizer_name,
            hybridize_flag,
            ctx
        )

        self.__learned_historical_df = self.__historical_df

        self.__batch_size = batch_size
        self.__agent_n = agent_n
        self.__defualt_money = defualt_money
        self.__rebalance_prob = rebalance_prob
        self.__rebalancing_prob_mean = rebalancing_prob_mean
        self.__rebalancing_prob_std = rebalancing_prob_std
        self.__risk_free_rate_mean = risk_free_rate_mean
        self.__risk_free_rate_std = risk_free_rate_std
        self.__agents_master_arr = agents_master_arr
        self.__rebalance_policy_list = rebalance_policy_list
        self.__rebalance_sub_policy_list = rebalance_sub_policy_list
        self.__timing_policy_list = timing_policy_list
        self.__historical_arr = historical_arr
        self.__stock_master_arr = stock_master_arr
        self.__next_action_n = next_action_n
        self.__first_date_key = first_date_key
        self.__ctx = ctx

        self.__deep_q_learning = deep_q_learning
        self.__q_logs_arr = None
        self.__logs_dir = logs_dir

    def sample_multi_agents(
        self,
        batch_size,
        start_date,
        end_date,
        date_fraction,
        agent_n,
        defualt_money,
        rebalancing_prob_mean=0.3,
        rebalancing_prob_std=0.01,
        risk_free_rate_mean=0.3,
        risk_free_rate_std=0.01,
    ):
        money_arr = np.ones((batch_size, agent_n, ))
        money_arr = money_arr * defualt_money
        rebalancing_prob_arr = np.random.normal(
            loc=rebalancing_prob_mean,
            scale=rebalancing_prob_std,
            size=(batch_size, agent_n, )
        )
        risk_free_rate_arr = np.random.normal(
            loc=risk_free_rate_mean,
            scale=risk_free_rate_std,
            size=(batch_size, agent_n, )
        )

        if rebalancing_prob_arr[rebalancing_prob_arr < 0.0].shape[0] > 0:
            raise ValueError("rebalancing probability must be more than 0.")
        if risk_free_rate_arr[risk_free_rate_arr < 0.0].shape[0] > 0:
            raise ValueError("`risk_free_rate_arr` must be more than 0.")
        if risk_free_rate_arr[risk_free_rate_arr > 1.0].shape[0] > 0:
            raise ValueError("`risk_free_rate_arr` must be less than 1.")

        agent_key_arr = None
        for batch in range(batch_size):
            _agent_key_arr = np.arange(agent_n).reshape((1, agent_n))
            if agent_key_arr is None:
                agent_key_arr = _agent_key_arr
            else:
                agent_key_arr = np.r_[agent_key_arr, _agent_key_arr]

        market_val_arr = np.zeros((batch_size, agent_n))
        buy_arr = np.zeros((batch_size, agent_n))
        sel_arr = np.zeros((batch_size, agent_n))
        cost_arr = np.zeros((batch_size, agent_n))
        income_gain_arr = np.zeros((batch_size, agent_n))

        agent_key_arr = np.expand_dims(agent_key_arr, axis=-1)
        money_arr = np.expand_dims(money_arr, axis=-1)
        market_val_arr = np.expand_dims(market_val_arr, axis=-1)
        rebalancing_prob_arr = np.expand_dims(rebalancing_prob_arr, axis=-1)
        risk_free_rate_arr = np.expand_dims(risk_free_rate_arr, axis=-1)
        buy_arr = np.expand_dims(buy_arr, axis=-1)
        sel_arr = np.expand_dims(sel_arr, axis=-1)
        cost_arr = np.expand_dims(cost_arr, axis=-1)
        income_gain_arr = np.expand_dims(income_gain_arr, axis=-1)

        agents_master_arr = np.concatenate(
            [
                agent_key_arr,
                money_arr,
                market_val_arr,
                rebalancing_prob_arr,
                risk_free_rate_arr,
                buy_arr,
                sel_arr,
                cost_arr,
                income_gain_arr,
            ],
            axis=-1
        )
        agents_master_arr = np.expand_dims(agents_master_arr, axis=-1)

        policy_list = [
            #"random_min_vol",
            #"random_max_sharpe_ratio",
            "min_vol",
            "max_sharpe_ratio",
        ]
        _rebalance_policy_list = ["min_vol", "max_sharpe_ratio"]

        if agent_n > 2:
            for _ in range(agent_n - 2):
                key = np.random.randint(low=0, high=len(policy_list))
                _rebalance_policy_list.append(policy_list[key])

        rebalance_policy_list = [_rebalance_policy_list for _ in range(batch_size)]

        sub_policy_list = [
            "buy_and_sell",
            "dollar_cost_averaging",
        ]
        _rebalance_sub_policy_list = [
            "buy_and_sell",
            "dollar_cost_averaging",
        ]
        if agent_n > 2:
            for _ in range(agent_n - 2):
                key = np.random.randint(low=0, high=len(sub_policy_list))
                _rebalance_sub_policy_list.append(sub_policy_list[key])

        rebalance_sub_policy_list = [_rebalance_sub_policy_list for _ in range(batch_size)]

        if self.__technical_flag is True:
            timing_policy_list = [
                "only_drl",
                "multi_trade_observer",
                #"bollinger_band_observer",
                #"contrarian_bollinger_band_observer",
                #"contrarian_rsi_observer",
                #"dive_timer",
                #"rsi_observer",
            ]
            _timing_policy_list = ["only_drl", "multi_trade_observer"]
        else:
            timing_policy_list = [
                "only_drl",
                #"multi_trade_observer",
                #"bollinger_band_observer",
                #"contrarian_bollinger_band_observer",
                #"contrarian_rsi_observer",
                #"dive_timer",
                #"rsi_observer",
            ]
            _timing_policy_list = ["only_drl", "only_drl"]

        if agent_n > 2:
            for n in range(agent_n - 2):
                if n > 2:
                    key = np.random.randint(low=0, high=len(timing_policy_list))
                else:
                    key = 0
                _timing_policy_list.append(timing_policy_list[key])

        timing_policy_list = [_timing_policy_list for _ in range(batch_size)]

        return (
            agents_master_arr, 
            rebalance_policy_list, 
            rebalance_sub_policy_list, 
            timing_policy_list,
            money_arr,
        )

    def build_dqn(
        self, 
        policy_sampler,
        learning_rate,
        epsilon_greedy_rate,
        alpha_value,
        gamma_value,
        initializer,
        optimizer_name,
        hybridize_flag,
        ctx
    ):
        computable_loss = L2NormLoss()

        output_nn = NeuralNetworks(
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # `list` of int` of the number of units in hidden/output layers.
            units_list=[1],
            # `list` of act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            activation_list=["sigmoid"],
            # `list` of `float` of dropout rate.
            dropout_rate_list=[0.0],
            # `list` of `mxnet.gluon.nn.BatchNorm`.
            hidden_batch_norm_list=[None],
            # `bool` for using bias or not in output layer(last hidden layer).
            output_no_bias_flag=True,
            # `bool` for using bias or not in all layer.
            all_no_bias_flag=True,
            # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            hybridize_flag=hybridize_flag,
            # `mx.gpu()` or `mx.cpu()`.
            ctx=ctx,
        )

        cnn = MobileNetV2(
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # is-a `mxnet.initializer.Initializer` for parameters of model. If `None`, it is drawing from the Xavier distribution.
            initializer=initializer,
            # `int` of the number of filters in input lauer.
            input_filter_n=32,
            # `tuple` or `int` of kernel size in input layer.
            input_kernel_size=(3, 3),
            # `tuple` or `int` of strides in input layer.
            input_strides=(1, 1),
            # `tuple` or `int` of zero-padding in input layer.
            input_padding=(1, 1),
            # `list` of information of bottleneck layers whose `dict` means ...
            # - `filter_rate`: `float` of filter expfilter.
            # - `filter_n`: `int` of the number of filters.
            # - `block_n`: `int` of the number of blocks.
            # - `stride`: `int` or `tuple` of strides.
            bottleneck_dict_list=[
                {
                    "filter_rate": 1,
                    "filter_n": 32,
                    "block_n": 1,
                    "stride": 1
                },
                {
                    "filter_rate": 1,
                    "filter_n": 32,
                    "block_n": 2,
                    "stride": 1
                },
            ],
            # `int` of the number of filters in hidden layers.
            hidden_filter_n=64,
            # `tuple` or `int` of pooling size in hidden layer.
            # If `None`, the pooling layer will not attatched in hidden layer.
            pool_size=None,
            # is-a `NeuralNetworks` or `mxnet.gluon.block.hybridblock.HybridBlock`.
            output_nn=output_nn,
            # `str` of name of optimizer.
            optimizer_name=optimizer_name,
            # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            hybridize_flag=hybridize_flag,
            # `mx.gpu()` or `mx.cpu()`.
            ctx=ctx,
        )

        function_approximator = FunctionApproximator(
            model=cnn, 
            initializer=initializer,
            hybridize_flag=hybridize_flag,
            scale=1.0, 
            ctx=ctx, 
        )

        DQN = DQNController(
            function_approximator=function_approximator,
            policy_sampler=policy_sampler,
            computable_loss=L2NormLoss(),
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            hybridize_flag=hybridize_flag,
            scale=1.0,
            ctx=ctx,
            initializer=initializer,
            recursive_learning_flag=True,
        )
        DQN.epsilon_greedy_rate = epsilon_greedy_rate
        DQN.alpha_value = alpha_value
        DQN.gamma_value = gamma_value

        return DQN

    def learning(
        self,
        iter_n=1000,
        transfer_flag=False,
        params_path="params/function_approximator.params",
    ):
        """
        Learn.

        Args:
            iter_n:                             `int` of the number of learning iteration.
            transfer_flag:                      `bool`. If `True`, this class will do transfer learning.
            params_path:                        `str` of path to function approximator's parameters.
        """
        self.__init_state_arr = self.__deep_q_learning.policy_sampler.state_arr.copy()
        # Execute learning.
        self.__logger.debug("Start learning...")

        if params_path is not None and transfer_flag is True:
            try:
                self.__deep_q_learning.function_approximator.load_parameters(params_path)
            except:
                self.__logger.debug("self.__deep_q_learning.function_approximator.load_parameters is false.")

        # Execute learning.
        self.__deep_q_learning.learn(
            # The number of searching.
            iter_n=iter_n,
        )
        self.__logger.debug("learning is end...")

        if params_path is not None:
            self.__deep_q_learning.function_approximator.save_parameters(params_path)

        self.__loss_arr = np.abs(
            self.__deep_q_learning.q_logs_arr[:, 0] - self.__deep_q_learning.q_logs_arr[:, 1]
        )

        self.__logger.debug("The mean of loss of Q-Values:")
        self.__logger.debug(self.__loss_arr.mean())

    def inferencing(self, limit=12):
        '''
        Inference.

        Args:
            limit:      `int` of the number of recursive inferences.
        
        Returns:
            Tuple data. The values are ...
                - `list` of agent's state. 
                - `list` of Q-value.
                - `list` of portfolio data. 
                - `list` of agent's last state. 
                - `list` of talkative num.
                - `list` of `str` of date. 
                - `list` of `str` of rebalance policy. 
                - `list` of `str` of sub-rebalance policy. 
                - `list` of `int` of unit of dollar cost. 
                - `np.array` of agent's money.

        '''
        self.__historical_df = self.__test_historical_df
        self.__generated_historical_df = self.__test_generated_historical_df

        self.__historical_df = self.__historical_df.sort_values(by=["date", "ticker"])
        self.__generated_historical_df = self.__generated_historical_df.sort_values(by=["date", "ticker"])

        generated_historical_df_list = [None] * self.__batch_size
        for batch in range(self.__batch_size):
            generated_historical_df_list[batch] = self.__generated_historical_df[
                self.__generated_historical_df.batch_n == str(batch)
            ]
        self.__generated_historical_df_list = generated_historical_df_list

        agents_master_arr, rebalance_policy_list, rebalance_sub_policy_list, timing_policy_list, money_arr = self.sample_multi_agents(
            self.__batch_size,
            self.__generated_start_date,
            self.__generated_end_date,
            self.__date_fraction,
            self.__agent_n,
            self.__defualt_money,
            rebalancing_prob_mean=self.__rebalancing_prob_mean,
            rebalancing_prob_std=self.__rebalancing_prob_std,
            risk_free_rate_mean=self.__risk_free_rate_mean,
            risk_free_rate_std=self.__risk_free_rate_std,
        )

        stock_master_df = pd.merge(
            left=self.__stock_master_df,
            right=self.__historical_df[self.__historical_df.date_key == 0][["ticker", "adjusted_close"]],
            on="ticker"
        )
        stock_master_df = stock_master_df.sort_values(by=["ticker"])

        try:
            stock_master_df["date"] = stock_master_df["date"]
            _stock_master_df = stock_master_df[stock_master_df.date_key == 0]
            stock_master_arr = _stock_master_df[[
                "adjusted_close", 
                "commission", 
                "tax", 
                "ticker", 
                "expense_ratio", 
                "asset_allocation",
                "area_allocation",
                "yield",
                "date",
                "asset",
                "area"
            ]].values
            daily_stock_master_flag = True
        except:
            stock_master_arr = stock_master_df[[
                "adjusted_close", 
                "commission", 
                "tax", 
                "ticker", 
                "expense_ratio", 
                "asset_allocation",
                "area_allocation",
                "yield",
                "asset",
                "area"
            ]].values
            daily_stock_master_flag = False

        historical_arr = self.__historical_df[[
            "date_key",
            "adjusted_close",
            "volume",
            "ticker",
        ]].values

        for i in range(len(self.__generated_historical_df_list)):
            self.__generated_historical_df_list[i] = self.__generated_historical_df_list[i][
                [
                    "date_key",
                    "adjusted_close",
                    "volume",
                    "ticker"
                ]
            ]

        self.__logger.info(
            "rebalance policy data:"
        )
        self.__logger.info(
            rebalance_policy_list[0]
        )
        self.__logger.info(
            "rebalance sub-policy data:"
        )
        self.__logger.info(
            rebalance_sub_policy_list[0]
        )

        policy_sampler = PortfolioPercentPolicy(
            batch_size=self.__batch_size,
            agents_master_arr=agents_master_arr,
            rebalance_policy_list=rebalance_policy_list,
            rebalance_sub_policy_list=rebalance_sub_policy_list,
            timing_policy_list=timing_policy_list,
            technical_observer_dict={
                "multi_trade_observer": MultiTradeObserver(
                    technical_observer_list=[
                        BollingerBandObserver(
                            time_window=20
                        ),
                        ContrarianBollingerBandObserver(
                            time_window=20
                        ),
                        ContrarianRSIObserver(
                            time_window=14,
                        ),
                        RSIObserver(
                            time_window=14,
                        ),
                        MACDObserver(
                            time_window=26*2,
                            long_term=26,
                            short_term=12,
                            signal_term=9,
                        ),
                    ]
                ),
            },
            historical_arr=historical_arr,
            stock_master_arr=stock_master_arr,
            rebalance_prob=self.__rebalance_prob,
            next_action_n=self.__next_action_n,
            first_date_key=self.__first_date_key,
            date_fraction=self.__date_fraction,
            generated_stock_df_list=self.__generated_historical_df_list,
            extractable_generated_historical_data=self.__extractable_generated_historical_data,
            ctx=self.__ctx,
            ticker_list=self.__ticker_list,
            start_date=self.__generated_start_date,
            end_date=self.__generated_end_date,
            daily_stock_master_flag=daily_stock_master_flag,
            stock_master_df=stock_master_df,
        )

        policy_sampler.fix_date_flag = True
        policy_sampler.update_last_flag = True

        self.__deep_q_learning.policy_sampler = policy_sampler
        self.__init_state_arr = policy_sampler.state_arr.copy()

        self.__now_date_key = 0

        self.__logger.debug("-" * 100)
        self.__logger.debug("Inferencing is started.")
        self.__logger.debug("-" * 100)

        state_arr_list = []
        q_value_arr_list = []
        portfolio_result_list = [[None] * self.__batch_size][0]
        agents_result_list = [[None] * self.__batch_size][0]
        talkative_num_list = [[None] * self.__batch_size][0]
        now_date_list = []

        state_arr = policy_sampler.state_arr
        try:
            for n in range(limit):
                if n == 0:
                    now_date = policy_sampler.now_date()
                    now_date_list.append(now_date)
                    self.__logger.info("Start inferencing in " + str(now_date))
                    agents_master_arr = policy_sampler.agents_master_arr.copy()
                    action_hold_arr = policy_sampler.action_hold_arr.copy()

                    for batch in range(self.__batch_size):
                        if portfolio_result_list[batch] is None:
                            portfolio_result_list[batch] = []
                        if agents_result_list[batch] is None:
                            agents_result_list[batch] = []
                        if talkative_num_list[batch] is None:
                            talkative_num_list[batch] = []

                        agents_result_list[batch].append(agents_master_arr[batch])
                        portfolio_result_list[batch].append(
                            policy_sampler.convert_map_into_portfolio(
                                state_arr[batch].asnumpy(), 
                                batch
                            )
                        )
                        talkative_num_list[batch].append(action_hold_arr[batch])

                possible_action_arr, action_meta_data_arr = policy_sampler.draw()

                next_q_arr = None
                possible_reward_value_arr = None
                next_q_arr = None
                possible_predicted_q_arr = None

                for possible_i in range(possible_action_arr.shape[1]):
                    if action_meta_data_arr is not None:
                        meta_data_arr = action_meta_data_arr[:, possible_i]
                    else:
                        meta_data_arr = None

                    # Inference Q-Values.
                    _predicted_q_arr = self.__deep_q_learning.function_approximator.inference(
                        possible_action_arr[:, possible_i]
                    )
                    if possible_predicted_q_arr is None:
                        possible_predicted_q_arr = nd.expand_dims(_predicted_q_arr, axis=1)
                    else:
                        possible_predicted_q_arr = nd.concat(
                            possible_predicted_q_arr,
                            nd.expand_dims(_predicted_q_arr, axis=1),
                            dim=1
                        )

                    # Observe reward values.
                    _reward_value_arr = policy_sampler.observe_reward_value(
                        state_arr, 
                        possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr,
                    )
                    if possible_reward_value_arr is None:
                        possible_reward_value_arr = nd.expand_dims(_reward_value_arr, axis=1)
                    else:
                        possible_reward_value_arr = nd.concat(
                            possible_reward_value_arr,
                            nd.expand_dims(_reward_value_arr, axis=1),
                            dim=1
                        )

                    # Inference the Max-Q-Value in next action time.
                    policy_sampler.observe_state(
                        state_arr=possible_action_arr[:, possible_i],
                        meta_data_arr=meta_data_arr
                    )

                    next_possible_action_arr, _ = policy_sampler.draw()
                    next_next_q_arr = None

                    for possible_j in range(next_possible_action_arr.shape[1]):
                        _next_next_q_arr = self.__deep_q_learning.function_approximator.inference(
                            next_possible_action_arr[:, possible_j]
                        )
                        if next_next_q_arr is None:
                            next_next_q_arr = nd.expand_dims(
                                _next_next_q_arr,
                                axis=1
                            )
                        else:
                            next_next_q_arr = nd.concat(
                                next_next_q_arr,
                                nd.expand_dims(
                                    _next_next_q_arr, 
                                    axis=1
                                ),
                                dim=1
                            )

                    next_max_q_arr = next_next_q_arr.max(axis=1)

                    if next_q_arr is None:
                        next_q_arr = nd.expand_dims(
                            next_max_q_arr,
                            axis=1
                        )
                    else:
                        next_q_arr = nd.concat(
                            next_q_arr,
                            nd.expand_dims(
                                next_max_q_arr,
                                axis=1
                            ),
                            dim=1
                        )

                # Select action.
                selected_tuple = self.__deep_q_learning.select_action(
                    possible_action_arr, 
                    possible_predicted_q_arr,
                    possible_reward_value_arr,
                    next_q_arr,
                    possible_meta_data_arr=action_meta_data_arr
                )
                action_arr, predicted_q_arr, reward_value_arr, next_q_arr, action_meta_data_arr = selected_tuple

                # Update State.
                state_arr, state_meta_data_arr = policy_sampler.update_state(
                    action_arr, 
                    meta_data_arr=action_meta_data_arr
                )

                policy_sampler.observe_state(
                    state_arr=state_arr,
                    meta_data_arr=state_meta_data_arr
                )

                state_arr_list.append(state_arr.asnumpy())
                q_value_arr_list.append(predicted_q_arr.asnumpy())

                now_date = policy_sampler.now_date()
                now_date_list.append(now_date)
                self.__logger.debug("Inference in " + str(now_date))

                agents_master_arr = policy_sampler.agents_master_arr.copy()
                action_hold_arr = policy_sampler.action_hold_arr.copy()

                for batch in range(self.__batch_size):
                    if portfolio_result_list[batch] is None:
                        portfolio_result_list[batch] = []
                    if agents_result_list[batch] is None:
                        agents_result_list[batch] = []
                    if talkative_num_list[batch] is None:
                        talkative_num_list[batch] = []

                    agents_result_list[batch].append(agents_master_arr[batch])
                    portfolio_result_list[batch].append(
                        policy_sampler.convert_map_into_portfolio(
                            state_arr[batch].asnumpy(), 
                            batch
                        )
                    )
                    talkative_num_list[batch].append(action_hold_arr[batch])

                # Check.
                end_flag = policy_sampler.check_the_end_flag(
                    state_arr, 
                    meta_data_arr=meta_data_arr
                )

                if end_flag is True:
                    self.__logger.debug("end flag.")
                    break

        except KeyboardInterrupt:
            self.__logger.debug("Keybord interrupte")

        try:
            policy_sampler.update_historical_data(None)
        except Exception as e:
            self.__logger.debug("last updating historical data is false.")
            self.__logger.debug(e)
        policy_sampler.update_last()

        agents_master_arr = policy_sampler.agents_master_arr.copy()
        action_hold_arr = policy_sampler.action_hold_arr.copy()

        now_date = policy_sampler.now_date()
        now_date_list.append(now_date)
        self.__logger.info("Inference in last date, " + str(now_date))

        for batch in range(self.__batch_size):
            portfolio_result_list[batch].append(
                policy_sampler.convert_map_into_portfolio(state_arr[batch].asnumpy(), batch)
            )
            agents_result_list[batch].append(agents_master_arr[batch])
            talkative_num_list[batch].append(action_hold_arr[batch])

        return (
            state_arr_list, 
            q_value_arr_list, 
            portfolio_result_list, 
            agents_result_list, 
            talkative_num_list, 
            now_date_list, 
            rebalance_policy_list, 
            rebalance_sub_policy_list, 
            timing_policy_list,
            money_arr
        )

    def get_loss_arr(self):
        return self.__loss_arr

    def get_q_logs_arr(self):
        return self.__q_logs_arr

    def get_learned_historical_df(self):
        return self.__learned_historical_df

    def get_historical_df(self):
        return self.__historical_df

    def get_generated_historical_df(self):
        return self.__generated_historical_df

    def set_readonly(self, value):
        raise TypeError()

    loss_arr = property(get_loss_arr, set_readonly)
    q_logs_arr = property(get_q_logs_arr, set_readonly)
    learned_historical_df = property(get_learned_historical_df, set_readonly)
    historical_df = property(get_historical_df, set_readonly)
    generated_historical_df = property(get_generated_historical_df, set_readonly)


def post_process_agent_master(agent_master_df, _portfolio_df, batch_size):

    first_date = None

    def calculate_date_n(v):
        g_s_d = datetime.strptime(first_date, "%Y-%m-%d")
        g_e_d = datetime.strptime(v, "%Y-%m-%d")
        diff_g = (g_e_d - g_s_d).days
        return diff_g

    batch_agent_master_df_list = []
    batch_portfolio_df_list = []
    pie_df_list = []

    for batch in range(batch_size):
        df = agent_master_df[agent_master_df["batch_n"] == str(batch)]
        portfolio_df = _portfolio_df[_portfolio_df["batch_n"] == str(batch)]

        first_date = df.date.values[0]

        df["cum_cost"] = 0.0
        df["cum_buy"] = 0.0
        df["cum_sell"] = 0.0

        df["sell"] = df["sell"].apply(float)
        df["buy"] = df["buy"].apply(float)
        df["cost"] = df["cost"].apply(float)
        df["assessment"] = df["assessment"].apply(float)
        df["money"] = df["money"].apply(float)
        df["date_n"] = df["date_n"].apply(int)

        df["trading_profit"] = df.sell - df.buy

        df_list = []
        for agent_key in df.agent_master_key.drop_duplicates().values:
            _df = df[df.agent_master_key == agent_key]
            _df.loc[(_df.agent_master_key == agent_key), "cum_cost"] = _df[_df.agent_master_key == agent_key].cost.cumsum()
            _df.loc[(_df.agent_master_key == agent_key), "cum_buy"] = _df[_df.agent_master_key == agent_key].buy.cumsum()
            _df.loc[(_df.agent_master_key == agent_key), "cum_sell"] = _df[_df.agent_master_key == agent_key].sell.cumsum()

            _df["cum_trading_profit_and_loss"] = _df.cum_sell - _df.cum_buy - _df.cum_cost
            _df["days"] = _df.date.apply(calculate_date_n)
            _df["yeild"] = 100 * (_df.cum_trading_profit_and_loss) / _df.cum_buy / (_df.days / 365)

            df_list.append(_df)

        df = pd.concat(df_list)
        batch_agent_master_df_list.append(df)

        agent_master_key_df = df[["agent_master_key"]].drop_duplicates()
        agent_master_key_df = agent_master_key_df.reset_index()

        iter_n_list = portfolio_df.iter_n.drop_duplicates().values.tolist()
        for iter_n in iter_n_list:
            _pie_df = portfolio_df[portfolio_df.iter_n == iter_n]
            _pie_df = pd.concat([agent_master_key_df.agent_master_key, _pie_df], axis=1)
            pie_df_list.append(_pie_df)

    return batch_agent_master_df_list, pie_df_list


import hashlib

def update_format_by_ticker(file_path, stock_master_df):
    ticker_str = "_".join(
        stock_master_df.ticker.drop_duplicates().sort_values().values.astype(str).tolist()
    )
    format_str = "." + file_path.split(".")[-1]
    after_format_str = "-" + hashlib.md5(ticker_str.encode()).hexdigest() + format_str
    file_path = file_path.replace(
        format_str,
        after_format_str
    )
    return file_path


def Main(params_dict):
    '''
    Entry points.
    
    Args:
        params_dict:    `dict` of parameters.
    '''
    from algowars.extractablehistoricaldata.extractable_generated_historical_data import ExtractableGeneratedHistoricalData
    import pandas as pd
    # not logging but logger.
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR
    import warnings

    warnings.resetwarnings()
    warnings.simplefilter('ignore', pd.core.common.SettingWithCopyWarning)

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
        logs_dir=params_dict["logs_dir"],
    )

    stock_master_df = pd.read_csv(params_dict["ticker_master_path"])

    try:
        stock_master_df.ticker = stock_master_df.ticker.astype(int).astype(str)
    except:
        stock_master_df.ticker = stock_master_df.ticker.astype(str)

    ticker_list = stock_master_df.ticker.drop_duplicates().values.tolist()

    if params_dict["stock_choiced"] != "all":
        if params_dict["stock_choiced"] == "random":
            ticker_arr = stock_master_df.ticker.drop_duplicates().values
            np.random.shuffle(ticker_arr)
            key = np.random.randint(low=5, high=ticker_arr.shape[0])
            ticker_list = ticker_arr[:key].tolist()
            stock_master_df = stock_master_df[stock_master_df.ticker.isin(ticker_list)]
        else:
            ticker_key_list = params_dict["stock_choiced"].split(",")
            stock_master_df = stock_master_df[stock_master_df.ticker.isin(ticker_key_list)]

    params_dict["g_params_path"] = update_format_by_ticker(
        params_dict["g_params_path"], 
        stock_master_df
    )
    params_dict["re_e_params_path"] = update_format_by_ticker(
        params_dict["re_e_params_path"], 
        stock_master_df
    )
    params_dict["d_params_path"] = update_format_by_ticker(
        params_dict["d_params_path"], 
        stock_master_df
    )
    params_dict["dqn_params_path"] = update_format_by_ticker(
        params_dict["dqn_params_path"], 
        stock_master_df
    )

    stock_master_df = stock_master_df[[
        col for col in stock_master_df.columns if col not in ["volume", "company_name"]
    ]]

    if stock_master_df.shape[0] < 2:
        raise ValueError("The number of ticker should be more than 1.")

    if params_dict["verbose"] is True:
        print("Build DQN Controller.")

    extractable_generated_historical_data = ExtractableGeneratedHistoricalData(
        extractable_historical_data=extractable_historical_data, 
        batch_size=params_dict["batch_size"],
        seq_len=params_dict["seq_len"],
        learning_rate=params_dict["learning_rate"],
        item_n=params_dict["item_n"],
        k_step=params_dict["k_step"],
        original_start_date=params_dict["start_date"],
        original_end_date=params_dict["end_date"],
        logs_dir=params_dict["logs_dir"],
        g_params_path=params_dict["g_params_path"],
        re_e_params_path=params_dict["re_e_params_path"],
        d_params_path=params_dict["d_params_path"],
        transfer_flag=params_dict["transfer_flag"],
        not_learning_flag=params_dict["not_learning_flag"],
        multi_fractal_mode=params_dict["multi_fractal_mode"],
        long_term_seq_len=params_dict["long_term_seq_len"]
    )

    portfolio_dqn_controller = PortfolioDQNController(
        date_fraction=params_dict["date_fraction"],
        extractable_historical_data=extractable_historical_data,
        stock_master_df=stock_master_df,
        start_date=params_dict["start_date"],
        end_date=params_dict["end_date"],
        learning_rate=params_dict["learning_rate_dqn"],
        batch_size=params_dict["batch_size"],
        seq_len=params_dict["seq_len"],
        channel=params_dict["agent_n"],
        generated_start_date=params_dict["generated_start_date"],
        generated_end_date=params_dict["generated_end_date"],
        rebalancing_prob_mean=params_dict["rebalancing_prob_mean"],
        rebalancing_prob_std=params_dict["rebalancing_prob_std"],
        defualt_money=params_dict["defualt_money"],
        risk_free_rate_mean=params_dict["risk_free_rate_mean"],
        risk_free_rate_std=params_dict["risk_free_rate_std"],
        rebalance_prob=0.3,
        next_action_n=params_dict["next_action_n"],
        first_date_key=params_dict["first_date_key"],
        agent_n=params_dict["agent_n"],
        epsilon_greedy_rate=params_dict["epsilon_greedy_rate"],
        alpha_value=params_dict["alpha_value"],
        gamma_value=params_dict["gamma_value"],
        logs_dir=params_dict["logs_dir"],
        result_dir=params_dict["result_dir"],
        extractable_generated_historical_data=extractable_generated_historical_data,
        transfer_flag=params_dict["transfer_flag"],
        technical_flag=params_dict["technical_flag"]
    )
    ticker_list = portfolio_dqn_controller.ticker_list

    if params_dict["check_tickers_only"] == 1:
        print(ticker_list)
        return

    if params_dict["verbose"] is True:
        print("Draw a sample from planed distributions.")

    if params_dict["not_learning_flag"] is False:
        try:
            portfolio_dqn_controller.learning(
                iter_n=params_dict["epoch"],
                transfer_flag=params_dict["transfer_flag"],
                params_path=params_dict["dqn_params_path"],
            )
            np.save(params_dict["result_dir"] + "loss_arr", portfolio_dqn_controller.loss_arr)
            np.save(params_dict["result_dir"] + "q_logs_arr", portfolio_dqn_controller.q_logs_arr)
        except KeyboardInterrupt:
            np.save(params_dict["result_dir"] + "loss_arr", portfolio_dqn_controller.loss_arr)
            np.save(params_dict["result_dir"] + "q_logs_arr", portfolio_dqn_controller.q_logs_arr)
            print("Interrupt and save the loss and Q-Values.")
            #return

    if params_dict["verbose"] is True:
        print("Execute inferencing.")

    key = "_".join(ticker_list)
    md5_hash = hashlib.md5(key.encode()).hexdigest()

    result_tuple = portfolio_dqn_controller.inferencing(limit=params_dict["limit"])
    state_arr_list, q_value_arr_list, portfolio_result_list, agents_result_list, talkative_num_list, now_date_list, rebalance_policy_list, rebalance_sub_policy_list, timing_policy_list, money_arr = result_tuple
    np.save(params_dict["result_dir"] + "state_arr_list_" + str(md5_hash), np.array(state_arr_list))
    np.save(params_dict["result_dir"] + "q_value_arr_list_" + str(md5_hash), np.array(q_value_arr_list))
    np.save(params_dict["result_dir"] + "portfolio_result_list_" + str(md5_hash), np.array(portfolio_result_list))
    np.save(params_dict["result_dir"] + "agents_result_list_" + str(md5_hash), np.array(agents_result_list))
    np.save(params_dict["result_dir"] + "talkative_num_list_" + str(md5_hash), np.array(talkative_num_list))
    np.save(params_dict["result_dir"] + "now_date_list_" + str(md5_hash), np.array(now_date_list))

    batch_agent_master_df_list = []
    for batch in range(params_dict["batch_size"]):
        agent_master_df_list = []
        for i in range(len(agents_result_list[batch])):
            agent_master_df = pd.DataFrame(
                agents_result_list[batch][i][:, :, 0],
                columns=[
                    "agent_master_key",
                    "money",
                    "assessment",
                    "rebalancing_prob",
                    "risk_free_rate",
                    "buy",
                    "sell",
                    "cost",
                    "income_gain",
                ]
            )
            rebalance_policy_df = pd.DataFrame(
                rebalance_policy_list[batch],
                columns=["rebalance_policy"]
            )
            rebalance_sub_policy_df = pd.DataFrame(
                rebalance_sub_policy_list[batch],
                columns=["rebalance_sub_policy"]
            )
            timing_policy_df = pd.DataFrame(
                timing_policy_list[batch],
                columns=["timing_policy"]
            )
            first_money_df = pd.DataFrame(
                money_arr[batch],
                columns=["first_money"]
            )

            agent_master_df = pd.concat(
                [
                    agent_master_df.reset_index(),
                    rebalance_policy_df.reset_index(),
                    rebalance_sub_policy_df.reset_index(),
                    pd.DataFrame(np.arange(rebalance_sub_policy_df.shape[0])),
                    first_money_df.reset_index(),
                    timing_policy_df.reset_index(),
                ],
                axis=1
            )
            agent_master_df["date"] = now_date_list[i]

            agent_master_df["first_money"] = agent_master_df["first_money"].apply(float)
            agent_master_df["iter_n"] = i
            agent_master_df["date_n"] = i * params_dict["date_fraction"]
            agent_master_df["batch_n"] = str(batch)
            agent_master_df_list.append(agent_master_df)

        agent_master_df = pd.concat(agent_master_df_list)
        batch_agent_master_df_list.append(agent_master_df)

    if params_dict["date_fraction"] > 2:
        for batch in range(len(batch_agent_master_df_list)):
            agent_master_df = batch_agent_master_df_list[batch]

            d_agent_master_df = agent_master_df[agent_master_df.rebalance_sub_policy == "dollar_cost_mean"]
            n_d_agent_master_df = agent_master_df[agent_master_df.rebalance_sub_policy != "dollar_cost_mean"]

            d_agent_key_list = d_agent_master_df.agent_master_key.drop_duplicates().values.tolist()
            df_list = []
            if len(d_agent_key_list) > 0:
                for i in range(len(d_agent_key_list)):
                    df = d_agent_master_df[d_agent_master_df.agent_master_key == d_agent_key_list[i]]
                    cum_buy = df.buy.apply(float).cumsum().values[-1]
                    cum_cost = df.cost.apply(float).cumsum().values[-1]
                    df["first_money"] = cum_buy + cum_cost
                    df["first_money"] = df["first_money"].apply(float)
                    df_list.append(df)

                d_agent_master_df = pd.concat(df_list)
                agent_master_df = pd.concat([d_agent_master_df, n_d_agent_master_df])

            agent_master_df = agent_master_df.sort_values(by=["agent_master_key", "iter_n"])
            agent_master_df["batch_n"] = str(batch)
            batch_agent_master_df_list[batch] = agent_master_df

    try:
        stock_master_df["date"] = stock_master_df["date"]
        stock_master_df = stock_master_df[stock_master_df.date_key == 0]
    except:
        pass

    try:
        stock_master_df.ticker = stock_master_df.ticker.astype(int).astype(str)
    except:
        stock_master_df.ticker = stock_master_df.ticker.astype(str)

    stock_master_df = stock_master_df[stock_master_df.ticker.isin(ticker_list)]

    batch_portfolio_df_list = []
    for batch in range(len(portfolio_result_list)):
        portfolio_list = portfolio_result_list[batch]
        portfolio_df_list = []
        for i in range(len(portfolio_list)):
            portfolio_arr = np.array(portfolio_list[i])
            portfolio_df = pd.DataFrame(portfolio_arr, columns=stock_master_df.ticker.dropna().values.tolist())
            portfolio_df["iter_n"] = i
            portfolio_df["date_n"] = i * params_dict["date_fraction"]
            portfolio_df["batch_n"] = str(batch)
            portfolio_df_list.append(portfolio_df)
        
        portfolio_df = pd.concat(portfolio_df_list)
        batch_portfolio_df_list.append(portfolio_df)

    agent_master_df = pd.concat(batch_agent_master_df_list)
    portfolio_df = pd.concat(batch_portfolio_df_list)
    batch_agent_master_df_list, pie_df_list = post_process_agent_master(
        agent_master_df,
        portfolio_df,
        len(portfolio_result_list)
    )
    agent_df = pd.concat(batch_agent_master_df_list)
    pie_df = pd.concat(pie_df_list)

    agent_df.to_csv(params_dict["result_dir"] + "result_agent_master_" + str(md5_hash) + ".csv", index=False)
    portfolio_df.to_csv(params_dict["result_dir"] + "result_portfolio_" + str(md5_hash) + ".csv", index=False)
    pie_df.to_csv(params_dict["result_dir"] + "pie_" + str(md5_hash) + ".csv", index=False)

    if params_dict["verbose"] is True:
        print("End.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run portfolio DQN.")
    
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
        default="logs/historical/",
        help="Logs dir."
    )

    parser.add_argument(
        "-rd",
        "--result_dir",
        type=str,
        default="result/",
        help="result dir."
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
        "-df",
        "--date_fraction",
        type=int,
        default=1,
        help="date fraction."
    )

    parser.add_argument(
        "-an",
        "--agent_n",
        type=int,
        default=10,
        help="The number of agent."
    )
    parser.add_argument(
        "-mm",
        "--defualt_money",
        type=float,
        default=12000,
        help="Each agent's default money."
    )
    parser.add_argument(
        "-rpm",
        "--rebalancing_prob_mean",
        type=float,
        default=1.0,
        help="The mean of each agent's probability(weight) of rebalancing."
    )
    parser.add_argument(
        "-rps",
        "--rebalancing_prob_std",
        type=float,
        default=0.01,
        help="The STD of each agent's probability(weight) of rebalancing."
    )
    parser.add_argument(
        "-rsm",
        "--risk_free_rate_mean",
        type=float,
        default=0.3,
        help="The mean of each agent's risk free rate."
    )
    parser.add_argument(
        "-rss",
        "--risk_free_rate_std",
        type=float,
        default=0.01,
        help="The STD of each agent's risk free rate."
    )

    parser.add_argument(
        "-ep",
        "--epoch",
        type=int,
        default=1000,
        help="The epoch."
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=5000,
        help="The iter of DQN(infer)."
    )

    parser.add_argument(
        "-nan",
        "--next_action_n",
        type=int,
        default=8,
        help="The number of next action."
    )

    parser.add_argument(
        "-fdk",
        "--first_date_key",
        type=int,
        default=1,
        help="The first date key."
    )

    parser.add_argument(
        "-sc",
        "--stock_choiced",
        type=str,
        default="all",
        help="`all`, `random`, or `x,y,z`."
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-04,
        help="Learnign rate for GAN."
    )

    parser.add_argument(
        "-lrdqn",
        "--learning_rate_dqn",
        type=float,
        default=1e-04,
        help="Learnign rate for DQN."
    )

    parser.add_argument(
        "-eps",
        "--epsilon_greedy_rate",
        type=float,
        default=0.7,
        help="Epsilon Greedy Rate."
    )

    parser.add_argument(
        "-alpha",
        "--alpha_value",
        type=float,
        default=1.0,
        help="Alpha value."
    )

    parser.add_argument(
        "-gamma",
        "--gamma_value",
        type=float,
        default=0.1,
        help="Gamma value."
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
        default=1,
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
        default=30,
        help="The length of sequences."
    )

    parser.add_argument(
        "-gpp",
        "--g_params_path",
        type=str,
        default="params/recursive_seq_2_seq_model_re_encoder_model.params",
        help="Path to your file of generator's learned params in ReSeq2Seq."
    )

    parser.add_argument(
        "-rpp",
        "--re_e_params_path",
        type=str,
        default="params/recursive_seq_2_seq_model_model.params",
        help="Path to your file of re-encoder's learned params in ReSeq2Seq."
    )

    parser.add_argument(
        "-dpp",
        "--d_params_path",
        type=str,
        default="params/discriminative_model_model.params",
        help="Path to your file of discriminator's learned params in ReSeq2Seq."
    )

    parser.add_argument(
        "-dqnpp",
        "--dqn_params_path",
        type=str,
        default="params/dqn_model.params",
        help="Path to your file of function-approximator's learned params in DQN."
    )

    parser.add_argument(
        "-tf",
        "--transfer_flag",
        action="store_true",
        default=False,
        help="Do transfer learning or not."
    )

    parser.add_argument(
        "-nlf",
        "--not_learning_flag",
        action="store_true",
        default=False,
        help="Do only inferecing or not."
    )

    parser.add_argument(
        "-mfm",
        "--multi_fractal_mode",
        action="store_true",
        default=True,
        help="Do R/S analysis and use H-exponent as loss function for GAN."
    )

    parser.add_argument(
        "-ltsl",
        "--long_term_seq_len",
        type=int,
        default=30,
        help="."
    )

    parser.add_argument(
        "-tmp",
        "--ticker_master_path",
        type=str,
        default="masterdata/ticker_master.csv",
        help="Path to ticker master data."
    )

    parser.add_argument(
        "-cto",
        "--check_tickers_only",
        type=int,
        default=0,
        help="0, 1."
    )

    parser.add_argument(
        "-tech",
        "--technical_flag",
        action="store_true",
        default=False,
        help="Use technical analysis or not."
    )

    args = parser.parse_args()
    params_dict = vars(args)
    print(params_dict)

    Main(params_dict)
