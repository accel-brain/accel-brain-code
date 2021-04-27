# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import random
import matplotlib.cm as cm


class DataVisualizer(object):
    '''
    Visualize the results of price data generation and portfolio optimization.
    '''

    def __init__(
        self, 
        master_data_path,
        display_max_columns=250,
        display_max_rows=50,
    ):
        """
        Init.

        Args:
            master_data_path:       `str` of the path to master data.
            display_max_columns:    `int` value. 
                                    Sets the maximum number of columns displayed 
                                    when a frame is pretty-printed.
            
            display_max_rows:       `int` value. 
                                    Sets the maximum number of rows displayed 
                                    when a frame is pretty-printed.

        """
        master_df = pd.read_csv(master_data_path)
        ticker_list = master_df.ticker.sort_values().values.tolist()

        key = "_".join(ticker_list)
        key = hashlib.md5(key.encode()).hexdigest()
        
        self.__master_df = master_df
        self.__ticker_list = ticker_list
        self.__key = key

        pd.set_option('display.max_columns', display_max_columns)
        pd.set_option('display.max_rows', display_max_rows)

    def plot_wave(self, mode="test", each_flag=False, random_noise_flag=False, random_noise_abs=0.01):
        '''
        Plot price data generated GAN.

        Args:
            mode:       `str` value.
                        - `train`: Plot the generated data for DQN's training.
                        - `test`: Plot the generated data for DQN's test.

            each_flag:  `bool` value.
                        If `True`, All patterns of all batches are output sequentially.

            random_noise_flag:  Add random noise or not.
                                You can add random noise (We assume uniform distribution: low=-1*random_noise_abs, high=random_noise_abs.) 
                                to both the original historical data and the generated historical data. 
            
            random_noise_abs:   `float` of absolute value of the random noise.
        '''
        key = self.__key
        h_df = pd.read_csv("result/historical_data_" + str(key) + ".csv")
        g_df = pd.read_csv("result/generated_historical_data_" + str(key) + ".csv")
        test_h_df = pd.read_csv("result/test_historical_data_" + str(key) + ".csv")
        test_g_df = pd.read_csv("result/test_generated_historical_data_" + str(key) + ".csv")

        if mode == "train":
            self.__plot_wave(h_df, g_df, each_flag=each_flag, random_noise_flag=random_noise_flag, random_noise_abs=random_noise_abs)
        elif mode == "test":
            self.__plot_wave(test_h_df, test_g_df, each_flag=each_flag, random_noise_flag=random_noise_flag, random_noise_abs=random_noise_abs)

    def __plot_wave(self, h_df, g_df, each_flag=False, random_noise_flag=False, random_noise_abs=0.01):
        for ticker in h_df.ticker.drop_duplicates().values:
            if each_flag is False:
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(1,1,1)
                _h_df = h_df[h_df.ticker == ticker]
                arr = _h_df.adjusted_close.values[:]
                if random_noise_flag is True:
                    arr = arr + np.random.uniform(low=-1*random_noise_abs, high=random_noise_abs, size=arr.shape)

                ax.plot(_h_df.date.values[:], arr, label="real historical data.")
            for batch in range(g_df.batch_n.max()):
                if each_flag is True:
                    fig = plt.figure(figsize=(20, 10))
                    ax = fig.add_subplot(1,1,1)
                    _h_df = h_df[h_df.ticker == ticker]
                    arr = _h_df.adjusted_close.values[:]
                    if random_noise_flag is True:
                        arr = arr + np.random.uniform(low=-1*random_noise_abs, high=random_noise_abs, size=arr.shape)

                    ax.plot(_h_df.date.values[:], arr, label="real historical data.")

                _g_df = g_df[g_df.batch_n == batch]
                _g_df = _g_df[_g_df.ticker == ticker]
                arr = _g_df.adjusted_close.values[:]
                if random_noise_flag is True:
                    arr = arr + np.random.uniform(low=-1*random_noise_abs, high=random_noise_abs, size=arr.shape)

                ax.plot(_g_df.date.values[:], arr, label="generated histroical data(batch: " + str(batch) + ").", linestyle="dashed")
                if each_flag is True:
                    ax.set_xticks([
                        _h_df.date.values[_h_df.shape[0]//5], 
                        _h_df.date.values[2*_h_df.shape[0]//5], 
                        _h_df.date.values[3*_h_df.shape[0]//5],
                        _h_df.date.values[4*_h_df.shape[0]//5],
                    ])
                    try:
                        company_name = self.__master_df[self.__master_df.ticker == ticker].company_name.values[0]
                    except:
                        company_name = ticker
                    plt.legend()
                    plt.title(company_name)
                    plt.show()

            if each_flag is False:
                ax.set_xticks([
                    _h_df.date.values[_h_df.shape[0]//5], 
                    _h_df.date.values[2*_h_df.shape[0]//5], 
                    _h_df.date.values[3*_h_df.shape[0]//5],
                    _h_df.date.values[4*_h_df.shape[0]//5],
                ])
                try:
                    company_name = self.__master_df[self.__master_df.ticker == ticker].company_name.values[0]
                except:
                    company_name = ticker
                plt.legend()
                plt.title(company_name)
                plt.show()

    def plot_portfolio(
        self, 
        rebalance_policy="min_vol", 
        rebalance_sub_policy=None, 
        timing_policy=None, 
        bad_end_flag=False
    ):
        '''
        Plot potrfolio optimized by DQN.

        Args:
            rebalance_policy:       `str` value.
                                    - `min_vol`: Visualize portfolios generated in a mode where portfolio optimization is based on volatility minimization.
                                    - `max_sharpe_ratio`: Visualize portfolios generated in a mode where portfolio optimization is based on sharpe ratio maximization.

            rebalance_sub_policy:   `str` value.
                                    - `buy_and_sell`: Outputs the results of agents performing normal trading.
                                    - `dollar_cost_averaging`: By referring to the dollar cost averaging method, the agent outputs the results when making an investment strategy decision (although it does not necessarily follow the dollar cost averaging method completely).

            timing_policy:          `str` value.
                                    - `only_drl`: The agent makes decisions on the timing of rebalancing and rotation by purely following the Q-value of deep reinforcement learning.
                                    - `multi_trade_observer`: The following agent results are output. By observing the technical indicator measuring instruments implemented in the interface of `MultiTradeObserver`, the agent can refer not only to the Q value of deep reinforcement learning but also to the" buy sign "and" sell sign "of various technical indicators. By doing so, you can make investment strategy decisions.

            bad_end_flag:           `bool` value. The default value is `False`.
                                    If `False`, this method outputs the result when the yield is the highest.
                                    If `True`, this method outputs the result when the yield is the lowest.
        '''
        key = self.__key
        ticker_n = self.__master_df.ticker.drop_duplicates().shape[0]
        col = cm.Spectral(np.arange(ticker_n)/ticker_n)
        result_agent_master_df = pd.read_csv("result/result_agent_master_" + str(key) + ".csv")

        result_agent_master_df = result_agent_master_df[result_agent_master_df.rebalance_policy == rebalance_policy]
        if rebalance_sub_policy is not None:
            result_agent_master_df = result_agent_master_df[result_agent_master_df.rebalance_sub_policy == rebalance_sub_policy]
        if timing_policy is not None:
            result_agent_master_df = result_agent_master_df[result_agent_master_df.timing_policy == timing_policy]

        try:
            if bad_end_flag is False:
                batch_n = result_agent_master_df[
                    result_agent_master_df.yeild == result_agent_master_df.yeild.max()
                ].batch_n.values[0]
            else:
                batch_n = result_agent_master_df[
                    result_agent_master_df.yeild == result_agent_master_df.yeild.min()
                ].batch_n.values[0]
                
        except IndexError:
            print("Data is not found.")
            return None

        agent_master_key = result_agent_master_df[result_agent_master_df.yeild == result_agent_master_df.yeild.max()].agent_master_key.values[0]

        talkative_num_list_arr = np.load("result/talkative_num_list_" + str(key) + ".npy")
        talkative_num_list_arr = talkative_num_list_arr.transpose((0, 2, 1, 3))
        talkative_num_arr = talkative_num_list_arr[int(batch_n), int(agent_master_key)]

        pie_df = pd.read_csv("result/pie_" + str(key) + ".csv")
        _pie_df = pie_df[pie_df.batch_n == batch_n]
        _pie_df = _pie_df[_pie_df.agent_master_key == agent_master_key]

        iter_n_arr = _pie_df.iter_n.drop_duplicates().values

        viz_pie_df = None
        for iter_n in iter_n_arr:
            try:
                df = _pie_df[_pie_df.iter_n == iter_n]
                date_n = df.date_n.values[0]
                date_str = result_agent_master_df[result_agent_master_df.days == date_n]["date"].values[0]
                label_list = [col for col in df.columns if col not in ["agent_master_key", "batch_n", "iter_n", "date_n"]]
            except:
                continue

            try:
                company_name_list = [self.__master_df[self.__master_df.ticker == ticker].company_name.values[0] for ticker in label_list]
            except:
                company_name_list = label_list
            
            pie_arr = df[label_list].values.reshape(-1, )
            pie_arr = pie_arr / pie_arr.sum()

            plt.figure(figsize=(8, 8))
            patches, texts, autotexts = plt.pie(
                pie_arr, 
                colors=col,
                labels=company_name_list, 
                counterclock=False, 
                startangle=90, 
                autopct=lambda p:'{:.1f}%'.format(p) if p>=0.1 else '',
                pctdistance=0.7,
            )
            plt.legend(
                company_name_list,
                loc=(0.62, 0.05),
                bbox_to_anchor=(1, 0, 0.5, 1),
            )
            plt.title(
                "date: " + date_str,
            )
            plt.show()
            if viz_pie_df is None:
                viz_pie_df = df[[col for col in df.columns if col not in ["iter_n", "date_n", "batch_n"]]]
            else:
                _viz_pie_df = df[[col for col in df.columns if col not in ["iter_n", "date_n", "batch_n"]]]
                viz_pie_df = pd.concat([viz_pie_df, _viz_pie_df])

        talkative_num_df = pd.DataFrame(talkative_num_arr)
        talkative_num_df.columns = [col + "_n" for col in viz_pie_df.columns if col != "agent_master_key"]

        df = result_agent_master_df[result_agent_master_df.batch_n == batch_n]
        df = df[df.agent_master_key == agent_master_key]
        df.money = df.money.astype(int)
        df.cost = df.cost.astype(int)
        df = df.sort_values(by=["agent_master_key", "date"])
        df = df.reset_index()
        viz_pie_df = viz_pie_df.reset_index()
        result_df = pd.concat(
            [
                df[
                    [
                        "date", 
                        "money", 
                        "assessment", 
                        "rebalancing_prob",
                        "buy", 
                        "cum_buy", 
                        "sell", 
                        "cum_sell", 
                        "cost", 
                        "cum_cost", 
                        "income_gain", 
                        "trading_profit", 
                        "cum_trading_profit_and_loss", 
                        "yeild",
                        "timing_policy",
                    ]
                ], 
                viz_pie_df,
                talkative_num_df
            ], 
            axis=1
        )
        return result_df
