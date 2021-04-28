# -*- coding: utf-8 -*-
from accelbrainbase.samplabledata.policy_sampler import PolicySampler
from accelbrainbase.iteratabledata.unlabeled_image_iterator import UnlabeledImageIterator

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss

from algowars.portfoliooptimization.volatility_minimization import VolatilityMinimization
from algowars.portfoliooptimization.sharpe_ratio_maximization import SharpeRatioMaximization
from algowars.exception.stock_history_error import StockHistoryError
from logging import getLogger
import mxnet.ndarray as nd
import mxnet as mx
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta


class PortfolioPercentPolicy(PolicySampler):
    '''
    Policy sampler for the Deep Q-learning to evaluate the value of the 
    "action" of selecting the image with the highest similarity based on 
    the "state" of observing an image.

    The state-action value is proportional to the similarity between the previously 
    observed image and the currently selected image.

    This class calculates the image similarity by mean squared error of images.
    '''
    # Now data.
    __now_date_key = 0
    # Date fraction.
    __date_fraction = 1
    
    # Multi-agent's rebalance policy.
    #   - mean_var:       To solve the Mean-variance minimalization problem.
    #   - sharpe_ratio:   To solve sharpe ratio maximalization problem.
    __rebalance_policy_list = []

    # rank-1 3D `np.ndarray` of historical data.
    #   - val1: Date key.
    #   - val2: Values(unit prices).
    #   - val3: Volumes.
    # This array must be sorted by `val1` and unique key of each stock.
    #__historical_arr = np.array([])
    
    # rank-1 and 4D `np.ndarray` of stock master data.
    #   - key:  Unique key of each stock.
    #   - val1: Unit prices in relation to the Market valuation.
    #   - val2: commission.
    #   - val3: Tax.
    #   - val4: Ticker.
    __stock_master_arr = np.array([])
    
    # `list` of Possible actions.
    __possible_action_list = [
        # not rebalance.
        0,
        # rebalance (In frist time step, this means build new portofolio.
        1
    ]
    
    # The problability of occurance of rebalancing.
    __rebalance_prob = 0.3
    
    # The number of sampled next actions.
    __next_action_n = 10
    
    # Memo of optimized result.
    __opt_memory = None

    __generated_historical_arr_list = None

    __fix_date_flag = False

    def get_fix_date_flag(self):
        ''' getter '''
        return self.__fix_date_flag
    
    def set_fix_date_flag(self, value):
        ''' setter '''
        self.__fix_date_flag = value

    fix_date_flag = property(get_fix_date_flag, set_fix_date_flag)

    __agents_master_arr_dict = {}

    __update_last_flag = False

    def get_update_last_flag(self):
        return self.__update_last_flag
    
    def set_update_last_flag(self, value):
        self.__update_last_flag = value

    update_last_flag = property(get_update_last_flag, set_update_last_flag)

    def __init__(
        self,
        batch_size,
        agents_master_arr,
        rebalance_policy_list,
        rebalance_sub_policy_list,
        timing_policy_list,
        technical_observer_dict,
        historical_arr,
        stock_master_arr,
        rebalance_prob=0.3,
        next_action_n=10,
        first_date_key=0,
        date_fraction=1,
        generated_stock_df_list=None,
        extractable_generated_historical_data=None,
        ctx=mx.gpu(),
        ticker_list=None,
        start_date=None,
        end_date=None,
        stock_master_df=None,
        daily_stock_master_flag=False,
    ):
        '''
        Initialize stock data.
        
        Args:
            batch_size:                                 `int` of batch size.
            agents_master_arr:                          `np.ndarray` of each multi-agents's money.
                                                        The shape is [batch, agent_i, dim].
                                                        The `dim` consists of the following items.
                                                        - agent key.
                                                        - money.
                                                        - market value.
                                                        - rebalancing probability.
                                                        - risk free rate.
                                                        - Sum of prices at the time of securities purchase.
                                                        - Sum of prices at the time of securities sale.
                                                        - Sum of costs paid.
                                                        - Sum of income gain.

            rebalance_policy_list:                      `list` of multi-agent's rebalance policy.
                                                        - `min_vol`:        To solve the volatility minimalization problem.
                                                        - `sharpe_ratio`:   To solve the sharpe ratio maximalization problem.

            rebalance_sub_policy_list:                  `list` of multi-agent's rebalance sub-policy.
                                                        - `buy_and_sell`: buy and sell.
                                                        - `buy_and_hold`: buy and hold.

            timing_policy_list:                         `list` of multi-agent's policy for timing of trades. For instance, ...
                                                        - `rsi_timer`: agent will decide timing of trades by thermodynamic the RSI.
                                                        - `bollinger_band_timer`: agent will decide timing of trades by the Bollinger bands.
                                                        - `dive_timer`: agent will decide timing of trades by thermodynamic dive.

            technical_observer_dict:                    `dict`. The key is a use-defined value of `timing_policy_list`.
                                                        The value is an object of `TechnicalObserver`.

            historical_arr:         rank-1 3D `np.ndarray` of historical data.
                                    - val1: Date key.
                                    - val2: Values(unit prices).
                                    - val3: Volumes.
                                    - val4: ticker.
                                    This array must be sorted by `val1` and unique key of each stock.
            
            stock_master_arr:       rank-1 and 5D `np.ndarray` of stock master data.
                                    - key:  Unique key of each stock.
                                    - val1: Unit prices in relation to the Market valuation.
                                    - val2: Commission.
                                    - val3: Tax.
                                    - val4: Ticker.
                                    - val5: Expense ratio.
                                    - val6: Asset allocation.
                                    - val7: Area allocation.

            rebalance_prob:         `float` of the problability of occurance of rebalancing.
            next_action_n:          `int` of the number of sampled next actions.
            first_date_key:         `int` of the first key of `true_stock_df` refered by this class.
            date_fraction:          `int` of date fraction.

            generated_stock_df_list:         `list` of `pd.DataFrame` of generated historical data.
                                                - val1: Date key.
                                                - val2: Values(unit prices).
                                                - val3: Volumes.
                                                - val4: ticker.
                                                This array must be sorted by `val1` and unique key of each stock.

            extractable_generated_historical_data:  is-a `ExtractableGeneratedHistoricalData`.

            ctx:                    `mx.cpu()` or `mx.gpu()`.
            ticker_list:            `list` of ticker symbols.
            start_date:             `str` of start date.
            end_date:               `str` of end date.
            stock_master_df:        `pd.DataFrame` of stock master data.
                                    - val1: Date key.
                                    - val2: Values(unit prices).
                                    - val3: Volumes.
                                    - val4: ticker.
                                    This array must be sorted by `val1` and unique key of each stock.

            daily_stock_master_flag:    `bool`. If `True`, this class considers the stock master data as variable data from day to day.
        '''
        self.__logger = getLogger("algowars")

        if isinstance(rebalance_policy_list, list) is False:
            raise TypeError("The type of `rebalance_policy_list` must be `list`.")
        if agents_master_arr.shape[0] != len(rebalance_policy_list):
            raise ValueError("The row of `agents_master_arr` and `rebalance_policy_list` is miss match.")
        if isinstance(rebalance_prob, float) is False:
            raise TypeError("The type of `rebalance_prob` must be `float`.")
        if isinstance(first_date_key, int) is False:
            raise TypeError("The type of `first_date_key` must be `int`.")
        
        if historical_arr[0].shape[0] < first_date_key:
            raise ValueError("`first_date_key` should be less than row of the `historical_arr`.")

        self.__possible_flag = False

        self.__portfolio_optimization_dict = {}
        self.__portfolio_optimization_dict.setdefault(
            "min_vol",
            VolatilityMinimization()
        )
        self.__portfolio_optimization_dict.setdefault(
            "max_sharpe_ratio",
            SharpeRatioMaximization()
        )

        self.__technical_observer_dict = technical_observer_dict

        self.__now_date_key = first_date_key
        self.__date_fraction = date_fraction
        self.__rebalance_policy_list = rebalance_policy_list
        self.__rebalance_sub_policy_list = rebalance_sub_policy_list
        self.__timing_policy_list = timing_policy_list

        self.__historical_arr = historical_arr

        self.__stock_master_arr_list = [stock_master_arr for _ in range(batch_size)]
        if daily_stock_master_flag is True and stock_master_df is not None:
            self.__stock_master_df_list = [stock_master_df for _ in range(batch_size)]

        self.__daily_stock_master_flag = daily_stock_master_flag

        self.__next_action_n  = next_action_n
        self.__agents_master_arr = agents_master_arr
        self.__opt_memo_dict_list = [{}] * batch_size
        self.__opt_memo_dict = {}

        if generated_stock_df_list is not None:
            self.__generated_historical_arr_list = [None] * len(generated_stock_df_list)
            for i in range(len(generated_stock_df_list)):
                self.__generated_historical_arr_list[i] = generated_stock_df_list[i].values

        self.__extractable_generated_historical_data = extractable_generated_historical_data

        self.t = 1
        self.__past_unit_price_df = None
        self.__ctx = ctx
        self.__ticker_list = ticker_list
        self.__start_date = start_date
        self.__end_date = end_date

        _start_date = datetime.strptime(start_date, "%Y-%m-%d")
        _end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.__total_days = (_end_date - _start_date).days

        portfolio_arr = np.zeros((
            batch_size,
            agents_master_arr.shape[0],
            stock_master_arr.shape[0]
        ))
        pre_portfolio_arr = portfolio_arr.copy()

        self.__state_hold_arr = None
        for batch in range(batch_size):
            if self.__state_hold_arr is None:
                self.__state_hold_arr = np.expand_dims(
                    self.__compute_default_hold(pre_portfolio_arr[batch], batch),
                    axis=0
                )
            else:
                self.__state_hold_arr = np.r_[
                    self.__state_hold_arr,
                    np.expand_dims(
                        self.__compute_default_hold(pre_portfolio_arr[batch], batch),
                        axis=0
                    )
                ]

        self.__real_state_hold_arr = self.__state_hold_arr.copy()

        self.action_hold_arr = None
        action_arr = None
        for batch in range(batch_size):
            for agent_i in range(agents_master_arr.shape[1]):
                try:
                    portfolio_arr[batch][agent_i] = self.__rebalance(
                        agent_i,
                        self.__rebalance_policy_list[batch][agent_i],
                        batch
                    )
                except ValueError as e:
                    print(e)
                    continue
                except StockHistoryError as e:
                    print(e)
                    continue

            if self.action_hold_arr is None:
                action_hold_arr = self.__compute_hold(portfolio_arr[batch], batch)
                action_hold_i_list = list(action_hold_arr.shape)
                action_hold_i_list.insert(0, batch_size)
                self.action_hold_arr = np.zeros(tuple(action_hold_i_list))

                self.action_hold_arr[batch] = action_hold_arr
            else:
                self.action_hold_arr[batch] = self.__compute_hold(portfolio_arr[batch], batch)

            if action_arr is None:
                _action_arr = self.__convert_portfolio_into_map(portfolio_arr[batch], batch)
                action_i_list = list(_action_arr.shape)
                action_i_list.insert(0, batch_size)
                action_arr = np.zeros(tuple(action_i_list))
                action_arr[batch] = _action_arr
            else:
                action_arr[batch] = self.__convert_portfolio_into_map(portfolio_arr[batch], batch)

        self.__cum_buy_arr = None
        self.__cum_sel_arr = None
        self.__cum_commission_tax_arr = None
        self.__cum_devided_ratio_arr = None

        self.state_arr = nd.ndarray.array(
            np.concatenate(
                [np.zeros_like(action_arr), np.ones_like(action_arr)],
                axis=1
            ),
            ctx=self.__ctx
        )

        action_arr = nd.ndarray.array(
            np.concatenate(
                [action_arr, np.ones_like(action_arr)],
                axis=1
            ),
            ctx=self.__ctx
        )
        self.__assessment_arr = None

        _ = self.observe_reward_value(self.state_arr, action_arr)

        self.__default_agents_master_arr = self.__agents_master_arr.copy()

        self.state_arr = nd.ndarray.array(
            action_arr,
            ctx=self.__ctx
        )
        self.__state_hold_arr = self.action_hold_arr.copy()

        self.__q_logs_arr = np.array([])
        self.__possible_observed_arr = None

    def draw(self):
        '''
        Draw samples from distribtions.
        
        Returns:
            `Tuple` of `mx.nd.array`s.
        '''
        possible_observed_arr, possible_label_arr = None, None
        for batch in range(self.__state_arr.shape[0]):
            portfolio_arr = self.__convert_map_into_portfolio(
                self.__state_arr[batch].asnumpy(),
                batch,
            )
            agents_master_arr = self.__agents_master_arr[batch]

            next_action_arr = np.empty((
                self.__next_action_n,
                agents_master_arr.shape[0],
                self.__stock_master_arr_list[batch].shape[0],
                100
            ))

            # if i = 0, not rebalance.
            # if i = 1, rebalance.
            # if i > 1, dicide by `TechnicalObserverDict`.
            for i in range(self.__next_action_n):
                portfolio_arr_ = portfolio_arr.copy()

                rebalance_flag = False
                for agent_i in range(agents_master_arr.shape[0]):
                    if portfolio_arr[agent_i].sum() == 0:
                        rebalance_flag = True
                    else:
                        if self.__rebalance_sub_policy_list[batch][agent_i] == "dollar_cost_averaging":
                            if self.t <= 1:
                                rebalance_flag = True
                            else:
                                rebalance_flag = False

                        elif i > 1:
                            if self.__generated_historical_arr_list is None or self.__possible_flag is False:
                                historical_arr = self.__historical_arr[
                                    self.__historical_arr[:, 0] <= self.__now_date_key
                                ]
                            else:
                                generated_historical_arr = self.__generated_historical_arr_list[batch][
                                    self.__generated_historical_arr_list[batch][:, 0] <= self.__now_date_key
                                ]
                                generated_historical_arr = generated_historical_arr[
                                    generated_historical_arr[:, 0] > (self.__now_date_key - self.__date_fraction)
                                ]
                                historical_arr = self.__historical_arr[
                                    self.__historical_arr[:, 0] <= (self.__now_date_key - self.__date_fraction)
                                ]

                                historical_arr = np.r_[historical_arr, generated_historical_arr]

                            if len(self.__technical_observer_dict) > 0 and self.__timing_policy_list[batch][agent_i] in self.__technical_observer_dict and self.__timing_policy_list[batch][agent_i] == "multi_trade_observer":
                                rebalance_weight_arr = self.__technical_observer_dict[self.__timing_policy_list[batch][agent_i]].decide_timing(
                                    historical_arr,
                                    agents_master_arr,
                                    portfolio_arr,
                                    self.__date_fraction,
                                    self.__ticker_list,
                                    agent_i
                                )
                            elif self.__timing_policy_list[batch][agent_i] == "only_drl":
                                rebalance_weight_arr = np.ones(len(self.__ticker_list))
                            else:
                                print(self.__technical_observer_dict)
                                print(self.__timing_policy_list[batch][agent_i])
                                raise ValueError("The value is `timing_policy_list` is invalid.")

                    # Rebalance.
                    if i == 1 or (i == 0 and rebalance_flag is True):
                        try:
                            try:
                                portfolio_arr_[agent_i] = self.__rebalance(
                                    agent_i,
                                    self.__rebalance_policy_list[batch][agent_i],
                                    batch
                                )
                            except StockHistoryError as e:
                                print(e)

                            portfolio_arr_[agent_i] = np.nan_to_num(portfolio_arr_[agent_i])
                            if portfolio_arr_[agent_i].min() < 0:
                                raise ValueError("The minimum value of portfolio_arr must be more than `0`.")

                        except ValueError as e:
                            self.__logger.debug("-" * 100)
                            self.__logger.debug("Rebalance is fault.")
                            self.__logger.debug(e)
                            self.__logger.debug("-" * 100)
                    elif i > 1:
                        portfolio_arr_[agent_i] = rebalance_weight_arr * portfolio_arr_[agent_i]
                        portfolio_arr_[agent_i] = portfolio_arr_[agent_i] / portfolio_arr_[agent_i].sum()

                if self.__possible_observed_arr is not None:
                    # (batch, action_n, agent_i, stock_i, 0-100%)
                    feature_arr = self.__convert_portfolio_into_map(
                        portfolio_arr_, 
                        batch,
                        self.__possible_observed_arr[batch][i],
                        check_zero_flag=True,
                    )
                else:
                    try:
                        feature_arr = self.__convert_portfolio_into_map(
                            portfolio_arr_, 
                            batch,
                            check_zero_flag=True,
                        )
                    except:
                        for _i in range(portfolio_arr_.shape[0]):
                            self.__logger.debug(portfolio_arr_[_i])
                            self.__logger.debug("")
                        raise

                next_action_arr[i] = feature_arr

            np.random.shuffle(next_action_arr)
            if possible_observed_arr is None:
                possible_observed_arr = np.expand_dims(next_action_arr, axis=0)
            else:
                possible_observed_arr = np.r_[
                    possible_observed_arr, 
                    np.expand_dims(next_action_arr, axis=0)
                ]
        
        for batch in range(possible_observed_arr.shape[0]):
            for action_n in range(possible_observed_arr.shape[1]):
                for agent_i in range(possible_observed_arr.shape[2]//2):
                    if possible_observed_arr[batch, action_n, agent_i].sum() == 0:
                        self.__logger.debug((batch, action_n, agent_i))
                        raise ValueError("checked.")
                    if possible_observed_arr[batch, action_n, agent_i].min() < 0:
                        self.__logger.debug((batch, action_n, agent_i))
                        raise ValueError("minus.")

        meta_arr = np.ones_like(possible_observed_arr) * self.__now_date_key
        observed_arr = nd.ndarray.array(possible_observed_arr, ctx=self.__state_arr.context)
        self.__batch_size = observed_arr.shape[0]

        if self.__possible_flag is False:
            self.__possible_agents_master_arr = self.__agents_master_arr.copy()
            self.__possible_cum_buy_arr = self.__cum_buy_arr.copy()
            self.__possible_cum_sel_arr = self.__cum_sel_arr.copy()
            self.__possible_cum_commission_tax_arr = self.__cum_commission_tax_arr.copy()
            self.__possible_cum_devided_ratio_arr = self.__cum_devided_ratio_arr.copy()
            if self.__assessment_arr is not None:
                self.__possible_assessment_arr = self.__assessment_arr.copy()
            else:
                self.__possible_assessment_arr = None

            self.__real_state_arr = self.state_arr
            if self.__state_hold_arr is None:
                self.__real_state_hold_arr = np.zeros_like(self.__state_hold_arr)
                for batch in range(self.__batch_size):
                    pre_portfolio_arr = self.__convert_map_into_portfolio(
                        self.state_arr[batch].asnumpy(),
                        batch
                    )
                    self.__real_state_hold_arr[batch] = self.__compute_hold(
                        pre_portfolio_arr,
                        batch,
                    )
            else:
                self.__real_state_hold_arr = self.__state_hold_arr.copy()

            self.__possible_state_hold_arr = self.__state_hold_arr.copy()

        self.__possible_flag = True
        self.__possible_observed_arr = possible_observed_arr

        return observed_arr, meta_arr

    def observe_reward_value(
        self, 
        state_arr, 
        action_arr,
        meta_data_arr=None,
    ):
        '''
        Compute the reward value.
        
        Args:
            state_arr:              Tensor of state.
            action_arr:             Tensor of action.
            meta_data_arr:          Meta data of actions.

        Returns:
            Reward value.
        '''
        if state_arr is None:
            state_arr = self.__state_arr

        batch_size = state_arr.shape[0]

        portfolio_arr = None
        for batch in range(batch_size):
            post_portfolio_arr = self.__convert_map_into_portfolio(
                action_arr[batch].asnumpy(),
                batch
            )

            self.action_hold_arr[batch] = self.__compute_hold(
                post_portfolio_arr,
                batch,
                self.__state_hold_arr[batch],
            )

            if portfolio_arr is None:
                portfolio_arr = np.expand_dims(post_portfolio_arr, axis=0)
            else:
                portfolio_arr = np.r_[
                    portfolio_arr,
                    np.expand_dims(post_portfolio_arr, axis=0)
                ]

        arr_tuple_list = self.__compute_invested_commission_tax()

        self.__buy_arr = None
        self.__sel_arr = None
        self.__commission_tax_arr = None
        self.__divided_ratio_arr = None
        for batch in range(len(arr_tuple_list)):
            if batch == 0:
                self.__buy_arr = np.expand_dims(arr_tuple_list[batch][0], axis=0)
                self.__sel_arr = np.expand_dims(arr_tuple_list[batch][1], axis=0)
                self.__commission_tax_arr = np.expand_dims(arr_tuple_list[batch][2], axis=0)
                self.__divided_ratio_arr = np.expand_dims(arr_tuple_list[batch][3], axis=0)
            else:
                self.__buy_arr = np.r_[
                    self.__buy_arr, 
                    np.expand_dims(arr_tuple_list[batch][0], axis=0)
                ]
                self.__sel_arr = np.r_[
                    self.__sel_arr,
                    np.expand_dims(arr_tuple_list[batch][1], axis=0)
                ]
                self.__commission_tax_arr = np.r_[
                    self.__commission_tax_arr,
                    np.expand_dims(arr_tuple_list[batch][2], axis=0)
                ]
                self.__divided_ratio_arr = np.r_[
                    self.__divided_ratio_arr,
                    np.expand_dims(arr_tuple_list[batch][3], axis=0)
                ]

        if self.__possible_flag is False:
            if self.__cum_buy_arr is not None:
                self.__cum_buy_arr += self.__buy_arr
            else:
                self.__cum_buy_arr = self.__buy_arr
            if self.__cum_sel_arr is not None:
                self.__cum_sel_arr += self.__sel_arr
            else:
                self.__cum_sel_arr = self.__sel_arr
            if self.__cum_commission_tax_arr is not None:
                self.__cum_commission_tax_arr += self.__commission_tax_arr
            else:
                self.__cum_commission_tax_arr = self.__commission_tax_arr
            if self.__cum_devided_ratio_arr is not None:
                self.__cum_devided_ratio_arr += self.__divided_ratio_arr
            else:
                self.__cum_devided_ratio_arr = self.__divided_ratio_arr
        else:
            self.__possible_cum_buy_arr += self.__buy_arr
            self.__possible_cum_sel_arr += self.__sel_arr
            self.__possible_cum_commission_tax_arr += self.__commission_tax_arr
            self.__possible_cum_devided_ratio_arr += self.__divided_ratio_arr

        reward_value_arr = None
        for batch in range(batch_size):
            if self.__possible_flag is False:
                post_assessment_arr = self.__compute_assessment(
                    self.__agents_master_arr[batch], 
                    self.action_hold_arr[batch],
                    batch
                )
            else:
                post_assessment_arr = self.__compute_assessment(
                    self.__possible_agents_master_arr[batch], 
                    self.action_hold_arr[batch],
                    batch
                )

            assessment_arr = np.nansum(
                post_assessment_arr.reshape(
                    (
                        post_assessment_arr.shape[0], 
                        -1
                    )
                ), 
                axis=1
            )

            if self.__possible_flag is False:
                if self.__assessment_arr is None:
                    assessment_i_list = list(assessment_arr.shape)
                    assessment_i_list.insert(0, batch_size)
                    self.__assessment_arr = np.zeros(tuple(assessment_i_list))

                self.__assessment_arr[batch] = assessment_arr
            else:
                if self.__possible_assessment_arr is None:
                    assessment_i_list = list(assessment_arr.shape)
                    assessment_i_list.insert(0, batch_size)
                    self.__possible_assessment_arr = np.zeros(tuple(assessment_i_list))

                self.__possible_assessment_arr[batch] = assessment_arr

            yeild_arr = self.__yeild_on_the_way(batch)
            yeild_arr = (yeild_arr - np.nanmean(yeild_arr)) / (yeild_arr.std() + 1e-08)

            if reward_value_arr is None:
                reward_value_arr = np.nansum(yeild_arr) / yeild_arr.shape[0]
                reward_value_arr = np.expand_dims(reward_value_arr, axis=0)
            else:
                reward_value_arr = np.r_[
                    reward_value_arr,
                    np.expand_dims(np.nansum(yeild_arr) / yeild_arr.shape[0], axis=0)
                ]

        if self.__possible_flag is False:
            self.__agents_master_arr[:, :, 1] -= np.expand_dims(self.__commission_tax_arr, axis=-1)
            self.__agents_master_arr[:, :, 1] -= np.expand_dims(self.__buy_arr, axis=-1)
            self.__agents_master_arr[:, :, 1] += np.expand_dims(self.__sel_arr, axis=-1)
            self.__agents_master_arr[:, :, 1] += np.expand_dims(self.__divided_ratio_arr, axis=-1)
            self.__agents_master_arr[:, :, 2] = np.expand_dims(self.__assessment_arr, axis=-1)
            self.__agents_master_arr[:, :, 5] = np.expand_dims(self.__buy_arr, axis=-1)
            self.__agents_master_arr[:, :, 6] = np.expand_dims(self.__sel_arr, axis=-1)
            self.__agents_master_arr[:, :, 7] = np.expand_dims(self.__commission_tax_arr, axis=-1)
            self.__agents_master_arr[:, :, 8] = np.expand_dims(self.__cum_devided_ratio_arr, axis=-1)
        else:
            self.__possible_agents_master_arr[:, :, 1] -= np.expand_dims(self.__commission_tax_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 1] -= np.expand_dims(self.__buy_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 1] += np.expand_dims(self.__sel_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 1] += np.expand_dims(self.__divided_ratio_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 2] = np.expand_dims(self.__possible_assessment_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 5] = np.expand_dims(self.__buy_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 6] = np.expand_dims(self.__sel_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 7] = np.expand_dims(self.__commission_tax_arr, axis=-1)
            self.__possible_agents_master_arr[:, :, 8] = np.expand_dims(self.__possible_cum_devided_ratio_arr, axis=-1)

        self.__commission_tax_arr = None
        self.__buy_arr = None
        self.__sel_arr = None

        reward_value_arr = nd.ndarray.array(reward_value_arr, ctx=state_arr.context)

        return reward_value_arr

    def get_action_hold_arr(self):
        return self.__action_hold_arr
    
    def set_action_hold_arr(self, value):
        self.__action_hold_arr = value
        if value is not None:
            for batch in range(self.__action_hold_arr.shape[0]):
                for agent_i in range(self.__action_hold_arr.shape[1]):
                    if self.__action_hold_arr[batch][agent_i].sum() == 0:
                        raise ValueError()

    action_hold_arr = property(get_action_hold_arr, set_action_hold_arr)

    def observe_state(self, state_arr, meta_data_arr):
        '''
        Observe states of agents in last epoch.

        Args:
            state_arr:      Tensor of state.
            meta_data_arr:  meta data of the state.
        '''
        self.state_arr = state_arr
        self.__state_meta_data_arr = meta_data_arr
        if self.__possible_flag is True:
            for batch in range(state_arr.shape[0]):
                pre_portfolio_arr = self.__convert_map_into_portfolio(
                    state_arr[batch].asnumpy(),
                    batch
                )
                self.__possible_state_hold_arr[batch] = self.__compute_hold(
                    pre_portfolio_arr,
                    batch,
                    self.action_hold_arr[batch]
                )

        # Update stock master data, refering the next day.
        # Update `__stock_master_arr`.
        try:
            self.__update_historical_data(meta_data_arr)
        except StockHistoryError:
            if self.fix_date_flag is True and self.update_last_flag is True:
                pass
            elif self.fix_date_flag is True and self.__possible_flag is True:
                self.__update_last()
            else:
                self.__reset_date_and_histroical_data()

    def __reset_date_and_histroical_data(self, retry_n=1):
        self.__now_date_key = np.random.randint(
            low=0,
            high=self.__date_fraction
        )
        self.__opt_memo_dict_list = [{}] * self.__batch_size
        self.__agents_master_arr = self.__default_agents_master_arr.copy()

        self.__buy_arr = None
        self.__sel_arr = None
        self.__commission_tax_arr = None
        self.__divided_ratio_arr = None
        self.__cum_buy_arr = np.zeros_like(self.__cum_buy_arr)
        self.__cum_sel_arr = np.zeros_like(self.__cum_sel_arr)
        self.__cum_commission_tax_arr = np.zeros_like(self.__cum_commission_tax_arr)
        batch_size = self.__batch_size
        portfolio_arr = np.zeros((
            batch_size,
            self.__agents_master_arr.shape[0],
            self.__stock_master_arr_list[0].shape[0]
        ))
        pre_portfolio_arr = portfolio_arr.copy()

        self.__real_state_hold_arr = self.__state_hold_arr.copy()

        self.action_hold_arr = None
        action_arr = None

        try:
            for batch in range(batch_size):
                for agent_i in range(self.__agents_master_arr.shape[1]):
                    portfolio_arr[batch][agent_i] = self.__rebalance(
                        agent_i,
                        self.__rebalance_policy_list[batch][agent_i],
                        batch
                    )
                    if portfolio_arr[batch][agent_i].sum() == 0:
                        raise ValueError("portfolio is zero.")

                if self.action_hold_arr is None:
                    action_hold_arr = self.__compute_hold(portfolio_arr[batch], batch)
                    action_hold_i_list = list(action_hold_arr.shape)
                    action_hold_i_list.insert(0, batch_size)
                    self.action_hold_arr = np.zeros(tuple(action_hold_i_list))

                    self.action_hold_arr[batch] = action_hold_arr
                else:
                    self.action_hold_arr[batch] = self.__compute_hold(portfolio_arr[batch], batch)

                if action_arr is None:
                    _action_arr = self.__convert_portfolio_into_map(portfolio_arr[batch], batch)
                    action_i_list = list(_action_arr.shape)
                    action_i_list.insert(0, batch_size)
                    action_arr = np.zeros(tuple(action_i_list))
                    action_arr[batch] = _action_arr
                else:
                    action_arr[batch] = self.__convert_portfolio_into_map(portfolio_arr[batch], batch)

        except ValueError:
            if retry_n < 10:
                self.__reset_date_and_histroical_data(retry_n=retry_n+1)
                return
            else:
                raise
        except StockHistoryError as e:
            if retry_n < 10:
                self.__reset_date_and_histroical_data(retry_n=retry_n+1)
                return
            else:
                raise

        self.__cum_buy_arr = None
        self.__cum_sel_arr = None
        self.__cum_commission_tax_arr = None
        self.__cum_devided_ratio_arr = None

        action_arr = nd.ndarray.array(
            np.concatenate(
                [action_arr, np.ones_like(action_arr)],
                axis=1
            ),
            ctx=self.__ctx
        )

        self.__assessment_arr = None
        _ = self.observe_reward_value(self.state_arr, action_arr)

        self.__default_agents_master_arr = self.__agents_master_arr.copy()

        self.state_arr = nd.ndarray.array(
            action_arr,
            ctx=self.__ctx
        )
        self.__state_hold_arr = self.action_hold_arr.copy()

        self.__possible_observed_arr = None

    def __check_debit(self):
        if self.__possible_flag is False:
            agents_master_arr = self.__agents_master_arr.copy()
        else:
            agents_master_arr = self.__possible_agents_master_arr.copy()

        agents_master_arr[:, :, 1] -= np.expand_dims(self.__commission_tax_arr, axis=-1)
        agents_master_arr[:, :, 1] -= np.expand_dims(self.__buy_arr, axis=-1)
        agents_master_arr[:, :, 1] += np.expand_dims(self.__sel_arr, axis=-1)
        agents_master_arr[:, :, 1] += np.expand_dims(self.__cum_devided_ratio_arr + self.__divided_ratio_arr, axis=-1)

        debit_tuple = np.where(agents_master_arr[:, :, 1] <= 0)
        if len(debit_tuple) == 3:
            batch_arr, agent_i_arr = debit_tuple[0], debit_tuple[1]
            if batch_arr.shape[0] > 0 and agent_i_arr.shape[0] > 0:
                self.__debit_tuple = batch_arr, agent_i_arr
                return batch_arr, agent_i_arr
            else:
                self.__debit_tuple = None
                return None
        else:
            self.__debit_tuple = None
            return None

    def update_state(
        self, 
        action_arr, 
        meta_data_arr=None
    ):
        '''
        Update state.
        
        This method can be overrided for concreate usecases.

        Args:
            action_arr:     action in `self.t`.
            meta_data_arr:  meta data of the action.
        
        Returns:
            Tuple data.
            - state in `self.t+1`.
            - meta data of the state.

            generated_stock_df_list:         `list` of `pd.DataFrame` of generated historical data.
                                                - val1: Date key.
                                                - val2: Values(unit prices).
                                                - val3: Volumes.
                                                - val4: ticker.
                                                This array must be sorted by `val1` and unique key of each stock.

        '''
        self.__possible_flag = False
        try:
            self.__switch_historical_data()
        except StockHistoryError:
            self.__reset_date_and_histroical_data()

        self.state_arr = self.__real_state_arr.copy()
        self.__state_hold_arr = self.__real_state_hold_arr.copy()

        _ = self.observe_reward_value(
            self.state_arr, 
            action_arr,
            meta_data_arr=meta_data_arr,
        )

        if self.__possible_flag is False:
            self.__state_hold_arr = self.action_hold_arr.copy()
        else:
            self.__possible_state_hold_arr = self.action_hold_arr.copy()

        self.t = self.t + 1
        return action_arr, meta_data_arr

    def check_the_end_flag(self, state_arr, meta_data_arr=None):
        '''
        Check the end flag.
        
        If this return value is `True`, the learning is end.

        As a rule, the learning can not be stopped.
        This method should be overrided for concreate usecases.

        Args:
            state_arr:      state in `self.t`.
            meta_data_arr:  meta data of the state.

        Returns:
            bool
        '''
        self.__possible_flag = False

        if self.fix_date_flag is False:
            return False

        # As a rule, the learning can not be stopped.
        next_date_key = self.__now_date_key + self.__date_fraction

        if self.__end_date is None:
            row = 0
            for add in range(10):
                row += self.__historical_arr[self.__historical_arr[:, 0] == next_date_key + add].shape[0]

            if row > 0:
                return False
            else:
                return True
        else:
            next_date_timestamp = (
                datetime.strptime(self.__start_date, "%Y-%m-%d") + timedelta(days=next_date_key)
            ).timestamp()
            end_date_timestamp = datetime.strptime(self.__end_date, "%Y-%m-%d").timestamp()
            if next_date_timestamp >= end_date_timestamp:
                return True
            else:
                return False

    def __now_date(self):
        _date = datetime.strptime(self.__start_date, "%Y-%m-%d") + timedelta(days=self.__now_date_key)
        _date = datetime.strftime(_date, "%Y-%m-%d")
        return _date

    def now_date(self):
        return self.__now_date()

    def __generated_date_range(self):
        generative_start_date = self.__now_date()
        generative_end_date = self.__end_date
        return generative_start_date, generative_end_date

    def __min_max_re_generated(self, _df, _target_arr, ticker_list):
        result_df_list = []
        col = "adjusted_close"
        for ticker in ticker_list:
            df = _df[_df.ticker == ticker]
            target_arr = _target_arr[_target_arr[:, 3] == ticker]
            _max, _min = target_arr[:, 1].max(), target_arr[:, 1].min()
            df[col] = df[col].fillna(0.0)
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            df[col] = (_max - _min) * df[col]
            df[col] = df[col] + _min
            result_df_list.append(df)

        result_df = pd.concat(result_df_list)
        result_df = result_df.sort_values(by=["date", "ticker"])
        return result_df

    def __re_generate_historical_data(self):
        max_date_key = self.__now_date_key

        start_date, end_date = self.__generated_date_range()
        not_learning_flag = self.__extractable_generated_historical_data.not_learning_flag
        self.__extractable_generated_historical_data.not_learning_flag = True
        generated_historical_df = self.__extractable_generated_historical_data.extract(
            start_date=start_date,
            end_date=end_date,
            ticker_list=self.__ticker_list
        )
        self.__extractable_generated_historical_data.not_learning_flag = not_learning_flag
        #generated_historical_df = generated_historical_df.sort_values(by=["batch_n", "date"])
        generated_historical_df = generated_historical_df.sort_values(by=["date", "ticker"])
        generated_historical_df = self.__min_max_re_generated(
            generated_historical_df,
            self.__historical_arr,
            self.__ticker_list
        )
        generated_historical_df.date_key = generated_historical_df.date_key + max_date_key

        generated_historical_df_list = [None] * self.__batch_size
        for batch in range(self.__batch_size):
            generated_historical_df_list[batch] = generated_historical_df[
                generated_historical_df.batch_n == str(batch)
            ]
            old_df = pd.DataFrame(self.__generated_historical_arr_list[batch])
            old_df.columns = [
                "date_key",
                "adjusted_close",
                "volume",
                "ticker"
            ]

            new_df_list = []
            for ticker in old_df.ticker.drop_duplicates().values:
                df = old_df[old_df.ticker == ticker]
                _max, _min = df.adjusted_close.max(), df.adjusted_close.min()
                df = generated_historical_df_list[batch]
                df = df[df.ticker == ticker]
                df.adjusted_close = (df.adjusted_close - df.adjusted_close.min()) / (df.adjusted_close.max() - df.adjusted_close.min())
                df.adjusted_close = (_max - _min) * df.adjusted_close
                df.adjusted_close = df.adjusted_close + _min
                new_df_list.append(df)
            generated_historical_df_list[batch] = pd.concat(new_df_list)
            generated_historical_df_list[batch] = generated_historical_df_list[batch].sort_values(by=["date", "ticker"])

        for i in range(len(generated_historical_df_list)):
            _df = generated_historical_df_list[i][
                [
                    "date_key",
                    "adjusted_close",
                    "volume",
                    "ticker"
                ]
            ]

            generated_historical_df_list[i] = _df.values

        self.__generated_historical_arr_list = generated_historical_df_list

    def __compute_default_hold(self, portfolio_arr, batch):
        hold_arr = np.zeros((
            self.__agents_master_arr[batch].shape[0],
            self.__stock_master_arr_list[batch].shape[0]
        ))

        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            hold_arr[agent_i] = portfolio_arr[agent_i]
            if self.__possible_flag is False:
                hold_arr[agent_i] *= self.__agents_master_arr[batch][agent_i, 1]
                hold_arr[agent_i] *= (1 - self.__agents_master_arr[batch][agent_i, 4])
            else:
                hold_arr[agent_i] *= self.__possible_agents_master_arr[batch][agent_i, 1]
                hold_arr[agent_i] *= (1 - self.__possible_agents_master_arr[batch][agent_i, 4])

            hold_arr[agent_i] = hold_arr[agent_i] / self.__stock_master_arr_list[batch][:, 0]

        hold_arr = np.ceil(hold_arr).astype(int)
        return hold_arr

    def __compute_hold(self, portfolio_arr, batch, pre_hold_arr=None):
        """
            agents_master_arr[batch, agent_i, dim]:      
                dim: rank-1 and 4D `np.ndarray` of each multi-agents's money.
                    agent_key_arr,
                    money_arr,
                    market_val_arr,
                    rebalancing_prob_arr,
                    risk_free_rate_arr,
                    buy_arr,
                    sel_arr,
                    cost_arr,
                    income_gain_arr,
        """
        hold_arr = np.zeros((
            self.__agents_master_arr[batch].shape[0],
            self.__stock_master_arr_list[batch].shape[0]
        ))
        commission_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        expense_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        divided_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        tax_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        buy_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        sel_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))

        if pre_hold_arr is not None:
            pre_hold_sum = pre_hold_arr.sum()
        else:
            pre_hold_sum = 0

        if self.__possible_flag is False:
            agents_master_arr = self.__agents_master_arr.copy()
            if pre_hold_arr is None or pre_hold_sum == 0:
                pre_state_hold_arr = self.__real_state_hold_arr[batch].copy()
            else:
                pre_state_hold_arr = pre_hold_arr.copy()
        else:
            agents_master_arr = self.__possible_agents_master_arr.copy()
            if pre_hold_arr is None or pre_hold_sum == 0:
                pre_state_hold_arr = self.__state_hold_arr[batch].copy()
            else:
                pre_state_hold_arr = pre_hold_arr.copy()

        now_date = datetime.strptime(self.__now_date(), "%Y-%m-%d")
        start_date = datetime.strptime(self.__start_date, "%Y-%m-%d")
        now_days = (now_date - start_date).days

        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            hold_arr[agent_i] = portfolio_arr[agent_i]

            if self.__possible_flag is False:
                # * (money_arr + market_val_arr)
                hold_arr[agent_i] *= (self.__agents_master_arr[batch][agent_i, 1] + self.__agents_master_arr[batch][agent_i, 2])

                # * (1 - risk_free_rate_arr)
                hold_arr[agent_i] *= (1 - self.__agents_master_arr[batch][agent_i, 4])
            else:
                # * (money_arr + market_val_arr)
                hold_arr[agent_i] *= (self.__possible_agents_master_arr[batch][agent_i, 1] + self.__possible_agents_master_arr[batch][agent_i, 2])

                # * (1 - risk_free_rate_arr)
                hold_arr[agent_i] *= (1 - self.__possible_agents_master_arr[batch][agent_i, 4])

            # / adjusted_close
            hold_arr[agent_i] = hold_arr[agent_i] / self.__stock_master_arr_list[batch][:, 0]
            # hold = portfolio_arr * (money_arr + market_val_arr) * (1 - risk_free_rate_arr) / adjusted_close

            if self.__rebalance_sub_policy_list[batch][agent_i] == "dollar_cost_averaging":
                hold_arr[agent_i] = hold_arr[agent_i] * now_days / self.__total_days

            hold_arr[agent_i] = np.ceil(hold_arr[agent_i]).astype(int)
            hold_arr[agent_i] = hold_arr[agent_i].astype(int)

            if hold_arr[agent_i].sum() == 0:
                hold_arr[agent_i] = pre_state_hold_arr[agent_i]
            else:
                limit = 100
                try_n = 0
                while True:
                    if try_n >= limit or np.max(hold_arr[agent_i]) <= 1:
                        hold_arr[agent_i] = pre_state_hold_arr[agent_i]
                        break

                    try_n = try_n + 1

                    commission_arr[agent_i] = np.abs(hold_arr[agent_i] - pre_state_hold_arr[agent_i]) * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 1]
                    expense_ratio_arr[agent_i] = hold_arr[agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 4].astype(float) * float(self.__date_fraction) / 365 / 100
                    divided_ratio_arr[agent_i] = hold_arr[agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 7].astype(float) * float(self.__date_fraction) / 365 / 100
                    buy_arr[agent_i] = np.maximum(hold_arr[agent_i] - pre_state_hold_arr[agent_i], 0) * self.__stock_master_arr_list[batch][:, 0]
                    sel_arr[agent_i] = np.maximum(pre_state_hold_arr[agent_i] - hold_arr[agent_i], 0) * self.__stock_master_arr_list[batch][:, 0]
                    tax_arr[agent_i] = np.maximum(sel_arr[agent_i] - buy_arr[agent_i], 0) * self.__stock_master_arr_list[batch][:, 2]

                    commission_tax = np.nansum(tax_arr[agent_i] + commission_arr[agent_i] + expense_ratio_arr[agent_i])
                    buy = buy_arr[agent_i].sum()
                    sel = sel_arr[agent_i].sum()

                    money = agents_master_arr[batch, agent_i, 1] - commission_tax
                    money = money - buy
                    money = money + sel

                    if money <= 0:
                        hold_arr[agent_i] = hold_arr[agent_i] / 2
                        hold_arr[agent_i] = np.maximum(hold_arr[agent_i], 0)
                        """
                        if hold_arr[agent_i].sum() <= 0:
                            hold_arr[agent_i] = pre_state_hold_arr[agent_i].copy()
                            break
                        """
                    else:
                        break

        hold_arr = np.maximum(hold_arr, 0)

        return hold_arr

    def __convert_portfolio_into_map(
        self, 
        portfolio_arr, 
        batch, 
        _portfolio_map_arr=None, 
        check_zero_flag=True,
        before_portfolio_arr=None
    ):
        portfolio_map_arr = np.zeros((
            self.__agents_master_arr[batch].shape[0],
            self.__stock_master_arr_list[batch].shape[0],
            100
        ))

        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            arr = portfolio_arr[agent_i].copy()
            S = np.nansum(portfolio_arr[agent_i])

            if S != 0:
                key_arr = portfolio_arr[agent_i] / S
                key_arr = 100 * key_arr
            else:
                raise ValueError("portfolio_arr[agent_i] is zero.")

            key_arr = key_arr.astype(int)
            if key_arr.max() == 100:
                key_arr = np.minimum(key_arr, 99)

            try:
                for stock_i in range(self.__stock_master_arr_list[batch].shape[0]):
                    if self.__possible_flag is False:
                        portfolio_map_arr[
                            agent_i, 
                            stock_i, 
                            key_arr[stock_i]
                        ] = portfolio_arr[agent_i][stock_i]
                    else:
                        portfolio_map_arr[
                            agent_i, 
                            stock_i, 
                            key_arr[stock_i]
                        ] = portfolio_arr[agent_i][stock_i]

            except IndexError:
                self.__logger.debug(arr)
                self.__logger.debug(portfolio_arr[agent_i])
                raise

            portfolio_map_arr[agent_i] = np.nan_to_num(portfolio_map_arr[agent_i])

            if portfolio_map_arr[agent_i].sum() == 0:
                if _portfolio_map_arr is not None:
                    portfolio_map_arr[agent_i] = _portfolio_map_arr[agent_i]
                else:
                    if check_zero_flag is True:
                        if before_portfolio_arr is None:
                            raise ValueError("The values of portfolio map are zeros.")
                        else:
                            arr = before_portfolio_arr[agent_i].copy()
                            S = np.nansum(before_portfolio_arr[agent_i])

                            if S != 0:
                                key_arr = before_portfolio_arr[agent_i] / S
                                key_arr = 100 * key_arr
                            else:
                                key_arr = before_portfolio_arr[agent_i]

                            key_arr = key_arr.astype(int)
                            if key_arr.max() == 100:
                                key_arr = np.minimum(key_arr, 99)

                            for stock_i in range(self.__stock_master_arr_list[batch].shape[0]):
                                if self.__possible_flag is False:
                                    portfolio_map_arr[
                                        agent_i, 
                                        stock_i, 
                                        key_arr[stock_i]
                                    ] = before_portfolio_arr[agent_i][stock_i]
                                else:
                                    portfolio_map_arr[
                                        agent_i, 
                                        stock_i, 
                                        key_arr[stock_i]
                                    ] = before_portfolio_arr[agent_i][stock_i]

        return portfolio_map_arr

    def convert_map_into_portfolio(self, portfolio_map_arr, batch):
        return self.__convert_map_into_portfolio(portfolio_map_arr, batch)

    def __convert_map_into_portfolio(self, portfolio_map_arr, batch, check_flag=True):
        portfolio_arr = np.zeros((
            self.__agents_master_arr[batch].shape[0],
            self.__stock_master_arr_list[batch].shape[0]
        ))
        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            if check_flag is True:
                if portfolio_map_arr[agent_i].sum() == 0:
                    self.__logger.debug(portfolio_map_arr.shape)
                    self.__logger.debug(agent_i)
                    raise ValueError("portfolio_map_arr is zero.")

            for stock_i in range(self.__stock_master_arr_list[batch].shape[0]):
                try:
                    #prob = np.where(portfolio_map_arr[agent_i, stock_i] > 0)[0][0]
                    prob = np.argmax(portfolio_map_arr[agent_i, stock_i])
                    portfolio_arr[agent_i, stock_i] = prob / 100
                except IndexError as e:
                    self.__logger.debug(e)
                    self.__logger.debug(("stock_i", stock_i))
                    continue

            if portfolio_arr[agent_i].sum() == 0:
                self.__logger.debug(portfolio_map_arr.shape)
                self.__logger.debug(portfolio_arr.shape)
                self.__logger.debug(agent_i)
                raise ValueError("portfolio_arr[agent_i] is zero.")

            portfolio_arr[agent_i] = portfolio_arr[agent_i] / np.nansum(portfolio_arr[agent_i])

        return portfolio_arr

    def __compute_assessment(self, agents_master_arr, hold_arr, batch):
        '''
            agents_master_arr:       rank-1 and 4D `np.ndarray` of each multi-agents's money.
                                    - val1: Multi-agents unique key.
                                    - val2: Money.
                                    - val3: Market valuation.
                                    - val4: The probability of rebalancing.
                                    - val5: Risk-free rate.
                                    - val6: Real invested values.
                                    - val7: Tax and commission.

        '''
        assessment_arr = np.empty((
            agents_master_arr.shape[0],
            self.__stock_master_arr_list[batch].shape[0]
        ))
        for agent_i in range(agents_master_arr.shape[0]):
            assessment_arr[agent_i] = hold_arr[agent_i] * self.__stock_master_arr_list[batch][:, 0]

        return assessment_arr

    def __compute_invested_commission_tax(
        self, 
    ):
        '''
            stock_master_arr:       rank-1 and 5D `np.ndarray` of stock master data.
                                    - key:  Unique key of each stock.
                                    - val1: Unit prices in relation to the Market valuation.
                                    - val2: Commission.
                                    - val3: Tax.
                                    - val4: Ticker.
                                    - val5: Expense ratio.
                                    - val6: Asset allocation.
                                    - val7: Area allocation.

        '''
        result_list = [None] * self.__agents_master_arr.shape[0]
        for batch in range(self.__agents_master_arr.shape[0]):
            commission_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            expense_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            divided_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            tax_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            buy_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            sel_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            for agent_i in range(self.__agents_master_arr[batch].shape[0]):
                commission_arr[agent_i] = np.abs(self.action_hold_arr[batch][agent_i] - self.__state_hold_arr[batch][agent_i]) * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 1]
                expense_ratio_arr[agent_i] = self.action_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 4].astype(float) * float(self.__date_fraction) / 365 / 100
                divided_ratio_arr[agent_i] = self.action_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 7].astype(float) * float(self.__date_fraction) / 365 / 100
                buy_arr[agent_i] = np.maximum(self.action_hold_arr[batch][agent_i] - self.__state_hold_arr[batch][agent_i], 0) * self.__stock_master_arr_list[batch][:, 0]
                sel_arr[agent_i] = np.maximum(self.__state_hold_arr[batch][agent_i] - self.action_hold_arr[batch][agent_i], 0) * self.__stock_master_arr_list[batch][:, 0]
                tax_arr[agent_i] = np.maximum(sel_arr[agent_i] - buy_arr[agent_i], 0) * self.__stock_master_arr_list[batch][:, 2]

            commission_tax_arr = np.nansum(tax_arr + commission_arr + expense_ratio_arr, axis=1)
            buy_arr = np.nansum(buy_arr, axis=1)
            sel_arr = np.nansum(sel_arr, axis=1)
            divided_ratio_arr = np.nansum(divided_ratio_arr, axis=1)

            result_list[batch] = (
                buy_arr, 
                sel_arr,
                commission_tax_arr,
                divided_ratio_arr,
            )

        return result_list

    def __rebalance(self, agent_i, opt_policy, batch):
        '''
        Rebalance.
        
        Args:
            opt_policy:        
            first_date_flag:  
        '''
        if self.__possible_flag is False:
            if (opt_policy, self.__now_date_key, int(self.__possible_flag)) in self.__opt_memo_dict:
                return self.__opt_memo_dict[(opt_policy, self.__now_date_key, int(self.__possible_flag))]
        else:
            if (opt_policy, self.__now_date_key, int(self.__possible_flag), batch) in self.__opt_memo_dict:
                return self.__opt_memo_dict[(opt_policy, self.__now_date_key, int(self.__possible_flag), batch)]

        """
        if self.__possible_flag is False:
            if (opt_policy, self.__now_date_key, int(self.__possible_flag)) in self.__opt_memo_dict_list[batch]:
                print((opt_policy, self.__now_date_key, int(self.__possible_flag)))
                return self.__opt_memo_dict_list[batch][(opt_policy, self.__now_date_key, int(self.__possible_flag))]
        else:
            if (opt_policy, self.__now_date_key, int(self.__possible_flag), batch) in self.__opt_memo_dict_list[batch]:
                print((opt_policy, self.__now_date_key, int(self.__possible_flag), batch))
                return self.__opt_memo_dict_list[batch][(opt_policy, self.__now_date_key, int(self.__possible_flag), batch)]
        """

        if self.__generated_historical_arr_list is None or self.__possible_flag is False:
            historical_arr = self.__historical_arr[
                self.__historical_arr[:, 0] <= self.__now_date_key
            ]
        else:
            generated_historical_arr = self.__generated_historical_arr_list[batch][
                self.__generated_historical_arr_list[batch][:, 0] <= self.__now_date_key
            ]
            generated_historical_arr = generated_historical_arr[
                generated_historical_arr[:, 0] > (self.__now_date_key - self.__date_fraction)
            ]
            historical_arr = self.__historical_arr[
                self.__historical_arr[:, 0] <= (self.__now_date_key - self.__date_fraction)
            ]

            if historical_arr.shape[0] > 0:
                historical_df = pd.DataFrame(historical_arr)
                historical_df.columns = [
                    "date_key",
                    "adjusted_close",
                    "volume",
                    "ticker",
                ]
                generated_historical_df = pd.DataFrame(generated_historical_arr)
                generated_historical_df.columns = [
                    "date_key",
                    "adjusted_close",
                    "volume",
                    "ticker",
                ]

                post_df_list = []
                for ticker in self.__ticker_list:
                    pre_df = historical_df[historical_df.ticker == ticker]
                    post_df = generated_historical_df[generated_historical_df.ticker == ticker]
                    try:
                        diff = post_df.adjusted_close.values[0] - pre_df.adjusted_close.values.mean()
                        post_df.adjusted_close = post_df.adjusted_close - diff
                    except Exception as e:
                        pass

                    post_df_list.append(post_df)

                generated_historical_arr = pd.concat(post_df_list).values

            historical_arr = np.r_[historical_arr, generated_historical_arr]

        asset_allocation_list = self.__stock_master_arr_list[batch][:, 8].tolist()
        asset_allocation_list = list(set(asset_allocation_list))
        area_allocation_list = self.__stock_master_arr_list[batch][:, 9].tolist()
        area_allocation_list = list(set(area_allocation_list))

        df = None
        for i in range(len(asset_allocation_list)):
            asset_stock_master_arr = self.__stock_master_arr_list[batch][self.__stock_master_arr_list[batch][:, 8] == asset_allocation_list[i]]
            for j in range(len(area_allocation_list)):
                stock_master_arr = asset_stock_master_arr[asset_stock_master_arr[:, 9] == area_allocation_list[j]]
                if stock_master_arr.shape[0] == 0:
                    continue

                ticker_list = stock_master_arr[:, 3].tolist()
                ticker_list = list(set(ticker_list))

                if len(ticker_list) > 1:
                    h_arr = None
                    for ticker in ticker_list:
                        arr = historical_arr[historical_arr[:, 3] == ticker]
                        arr = arr[:, 1]
                        if h_arr is None:
                            h_arr = historical_arr[historical_arr[:, 3] == ticker]
                        else:
                            h_arr = np.r_[h_arr, historical_arr[historical_arr[:, 3] == ticker]]

                    if self.__possible_flag is False:
                        h_df = self.__portfolio_optimization_dict[opt_policy].optimize(
                            h_arr,
                            risk_free_rate=self.__agents_master_arr[batch][agent_i, 4][0]
                        )
                    else:
                        h_df = self.__portfolio_optimization_dict[opt_policy].optimize(
                            h_arr,
                            risk_free_rate=self.__possible_agents_master_arr[batch][agent_i, 4][0]
                        )
                else:
                    h_df = pd.DataFrame(
                        [1.0], 
                        columns=[ticker_list[0]],
                        index=['allocation']
                    )

                for ticker in h_df.columns:
                    arr = stock_master_arr[stock_master_arr[:, 3] == ticker]
                    h_df[ticker] = h_df[ticker] * float(arr[:, 5])
                    h_df[ticker] = h_df[ticker] * float(arr[:, 6])

                if df is None:
                    df = h_df
                else:
                    df = pd.concat([df, h_df], axis=1)

        if df.dropna().shape[0] == 0:
            past_date_key = self.__now_date_key - self.__date_fraction
            while (opt_policy, past_date_key, int(self.__possible_flag)) not in self.__opt_memo_dict_list[batch]:
                past_date_key -= 1
                if past_date_key < 0:
                    raise ValueError()

            portfolio_arr = self.__opt_memo_dict_list[batch][(opt_policy, past_date_key, int(self.__possible_flag))]
        else:
            portfolio_arr = df[self.__stock_master_arr_list[batch][:, 3].tolist()].values.T.reshape(-1, )

        portfolio_arr = portfolio_arr / np.nansum(portfolio_arr)

        if portfolio_arr.sum() == 0:
            raise ValueError("Rebalanced portfolio is zero.")

        if self.__possible_flag is False:
            self.__opt_memo_dict.setdefault(
                (opt_policy, self.__now_date_key, int(self.__possible_flag)),
                portfolio_arr
            )
        else:
            self.__opt_memo_dict.setdefault(
                (opt_policy, self.__now_date_key, int(self.__possible_flag), batch),
                portfolio_arr
            )

        """
        if self.__possible_flag is False:
            self.__opt_memo_dict_list[batch].setdefault(
                (opt_policy, self.__now_date_key, int(self.__possible_flag)), 
                portfolio_arr
            )
        else:
            self.__opt_memo_dict_list[batch].setdefault(
                (opt_policy, self.__now_date_key, int(self.__possible_flag), batch), 
                portfolio_arr
            )
        """

        return portfolio_arr

    def update_historical_data(self, date_key_arr=None):
        self.__update_historical_data(date_key_arr)

    def __update_historical_data(self, date_key_arr=None):
        '''
        Update historical data.

        todo: possible_flag is True,  stock_master_arr もバッチ化
        '''
        if date_key_arr is None:
            self.__now_date_key += self.__date_fraction
        else:
            self.__now_date_key = date_key_arr.max() + self.__date_fraction
        
        self.__switch_historical_data()
        #self.__re_generate_historical_data()

    def __switch_historical_data(self):
        for batch in range(self.__batch_size):
            for add in range(10):
                if self.__possible_flag is False:
                    new_arr = self.__historical_arr[self.__historical_arr[:, 0] == self.__now_date_key + add]
                else:
                    new_arr = self.__generated_historical_arr_list[batch][
                        self.__generated_historical_arr_list[batch][:, 0] == self.__now_date_key + add
                    ]

                if self.__daily_stock_master_flag is True:
                    master_df = self.__stock_master_df_list[batch][
                        self.__stock_master_df_list[batch]["date_key"] == self.__now_date_key + add
                    ]
                    if master_df.shape[0] > 0 and new_arr.shape[0] > 0:
                        break
                else:
                    if new_arr.shape[0] > 0:
                        break

            if new_arr.shape[0] == 0:
                raise StockHistoryError("new arr is zero.")
            elif self.__daily_stock_master_flag is True and master_df.shape[0] == 0:
                raise StockHistoryError("new master df is zero.")
            else:
                if self.__daily_stock_master_flag is False:
                    old_master_arr = self.__stock_master_arr_list[batch].copy()
                    new_master_val_list = []
                    for i in range(new_arr.shape[0]):
                        new_master_list = self.__stock_master_arr_list[
                            batch
                        ][
                            self.__stock_master_arr_list[batch][:, 3] == new_arr[i, 3]
                        ].tolist()[0]

                        past = new_master_list[0]
                        new_master_list[0] = new_arr[i, 1]
                        new = new_master_list[0]
                        new_master_val_list.append(new_master_list)

                    new_master_arr = np.array(new_master_val_list)
                    new_master_arr[:, 0] = new_master_arr[:, 0].astype(np.float)
                    new_master_arr[:, 1] = new_master_arr[:, 1].astype(np.float)
                    new_master_arr[:, 2] = new_master_arr[:, 2].astype(np.float)

                    df = pd.DataFrame(
                        np.r_[new_master_arr, old_master_arr],
                        columns=[
                            "unit_price", 
                            "commission", 
                            "tax", 
                            "ticker", 
                            "expense_ratio", 
                            "asset_allocation",
                            "area_allocation",
                            "yield",
                            "asset", 
                            "area"
                        ]
                    )
                    df = df.drop_duplicates(["ticker"])

                else:
                    old_master_arr = master_df[[
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

                    df = pd.DataFrame(
                        np.c_[new_arr[:, 1], old_master_arr],
                        columns=[
                            "unit_price", 
                            "commission", 
                            "tax", 
                            "ticker", 
                            "expense_ratio", 
                            "asset_allocation",
                            "area_allocation",
                            "yield",
                            "asset", 
                            "area"
                        ]
                    )

            df["unit_price"] = df["unit_price"].astype(float)
            df["commission"] = df["commission"].astype(float)
            df["tax"] = df["tax"].astype(float)

            self.__past_unit_price_df = pd.DataFrame(
                self.__historical_arr[
                    self.__historical_arr[:, 0] < self.__now_date_key
                ][-self.__date_fraction:]
            )
            self.__past_unit_price_df.columns = [
                "_",
                "unit_price",
                "__",
                "ticker",
            ]
            self.__past_unit_price_df = self.__past_unit_price_df[["unit_price", "ticker"]]
            self.__past_unit_price_df.unit_price = self.__past_unit_price_df.unit_price.astype(float)

            self.__stock_master_arr_list[batch] = df.values

    def update_last(self):
        self.__update_last()

    def __update_last(self):
        for batch in range(self.__agents_master_arr.shape[0]):
            commission_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            tax_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            expense_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            divided_ratio_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
            sel_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))

            for agent_i in range(self.__agents_master_arr[batch].shape[0]):
                commission_arr[agent_i] = self.__state_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 1]
                tax_arr[agent_i] = self.__state_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 2]
                expense_ratio_arr[agent_i] = self.__state_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 4].astype(float) * float(self.__date_fraction) / 365 / 100
                divided_ratio_arr[agent_i] = self.__state_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 7].astype(float) * float(self.__date_fraction) / 365 / 100
                sel_arr[agent_i] = self.__state_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0]

            commission_tax_arr = np.nansum(tax_arr + commission_arr, axis=1)
            sel_arr = np.nansum(sel_arr, axis=1)
            divided_ratio_arr = np.nansum(divided_ratio_arr, axis=1)

            for agent_i in range(self.__agents_master_arr[batch].shape[0]):
                self.__agents_master_arr[batch][agent_i, 1] -= commission_tax_arr[agent_i]
                self.__agents_master_arr[batch][agent_i, 1] += sel_arr[agent_i]
                self.__agents_master_arr[batch][agent_i, 1] += divided_ratio_arr[agent_i]
                self.__agents_master_arr[batch][agent_i, 2] = 0.0
                self.__agents_master_arr[batch][agent_i, 5] = 0.0
                self.__agents_master_arr[batch][agent_i, 6] += sel_arr[agent_i]
                self.__agents_master_arr[batch][agent_i, 7] += commission_tax_arr[agent_i]
                self.__agents_master_arr[batch][agent_i, 8] = self.__cum_devided_ratio_arr[agent_i].sum() + divided_ratio_arr[agent_i]

    def __yeild_on_the_way(self, batch):
        # shape is not changed, even if possible or not.
        commission_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        tax_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        buy_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))
        sel_arr = np.zeros((self.__agents_master_arr[batch].shape[0], self.__stock_master_arr_list[batch].shape[0]))

        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            commission_arr[agent_i] = self.action_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0] * self.__stock_master_arr_list[batch][:, 1]
            sel_arr[agent_i] = self.action_hold_arr[batch][agent_i] * self.__stock_master_arr_list[batch][:, 0]

        commission_tax_arr = np.nansum(tax_arr + commission_arr, axis=1)
        sel_arr = np.nansum(sel_arr, axis=1)

        if self.__possible_flag is False:
            cum_trading_profit_and_loss_arr = sel_arr + self.__cum_sel_arr[batch] - self.__cum_buy_arr[batch] - self.__cum_commission_tax_arr[batch] - commission_tax_arr
        else:
            cum_trading_profit_and_loss_arr = sel_arr + self.__possible_cum_sel_arr[batch] - self.__possible_cum_buy_arr[batch] - self.__possible_cum_commission_tax_arr[batch] - commission_tax_arr

        yeild_arr = np.zeros(self.__agents_master_arr[batch].shape[0])
        for agent_i in range(self.__agents_master_arr[batch].shape[0]):
            yeild = 100 * (cum_trading_profit_and_loss_arr[agent_i])

            if self.__possible_flag is False:
                yeild = yeild / (1e-08 + self.__cum_buy_arr[batch][agent_i] + self.__cum_commission_tax_arr[batch][agent_i])
            else:
                yeild = yeild / (1e-08 + self.__possible_cum_buy_arr[batch][agent_i] + self.__possible_cum_commission_tax_arr[batch][agent_i])

            yeild = yeild / ((((self.t * self.__date_fraction) + 1) / 365) + 1)
            yeild_arr[agent_i] = yeild

        return yeild_arr

    def get_agents_master_arr(self):
        return self.__agents_master_arr
    
    def set_agents_master_arr(self, value):
        self.__agents_master_arr = value
    
    agents_master_arr = property(get_agents_master_arr, set_agents_master_arr)

    def get_action_hold_arr(self):
        return self.__action_hold_arr

    def set_action_hold_arr(self, value):
        self.__action_hold_arr = value

    action_hold_arr = property(get_action_hold_arr, set_action_hold_arr)

    def get_next_action_n(self):
        return self.__next_action_n
    
    def set_next_action_n(self, value):
        self.__next_action_n = value

    next_action_n = property(get_next_action_n, set_next_action_n)

    def get_now_date_key(self):
        return self.__now_date_key
    
    def set_now_date_key(self, value):
        self.__now_date_key = value

    now_date_key = property(get_now_date_key, set_now_date_key)

    def get_historical_arr(self):
        return self.__historical_arr
    
    def set_historical_arr(self, value):
        self.__historical_arr = value

    historical_arr = property(get_historical_arr, set_historical_arr)

    def set_readonly(self, value):
        ''' setter '''
        raise TypeError("This property must be read-only.")

    def set_state_arr(self, value):
        if isinstance(value, nd.NDArray) is False:
            raise TypeError()
        if isinstance(value, tuple) is True:
            raise TypeError()

        self.__state_arr = value

    def get_state_arr(self):
        ''' setter '''
        return self.__state_arr

    state_arr = property(get_state_arr, set_state_arr)

    def get_q_logs_arr(self):
        ''' getter '''
        return self.__q_logs_arr
    
    def set_q_logs_arr(self, values):
        ''' setter '''
        raise TypeError("The `q_logs_arr` must be read-only.")
    
    q_logs_arr = property(get_q_logs_arr, set_q_logs_arr)
