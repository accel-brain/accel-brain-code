# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date, datetime, timedelta

from algowars.extractable_historical_data import ExtractableHistoricalData
from algowars.truesampler.volatility_conditional_true_sampler import VolatilityConditionalTrueSampler
from algowars.noisesampler.volatility_conditional_noise_sampler import VolatilityConditionalNoiseSampler
from algowars.generativemodel.recursive_seq2seq_model import RecursiveSeq2SeqModel
from algowars.generativemodel.recursiveseq2seqmodel.multi_fractal_seq2seq_model import MultiFractalSeq2SeqModel

from algowars.controllablemodel.gancontroller.volatility_gan_controller import VolatilityGANController

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutionalautoencoder.convolutional_ladder_networks import ConvolutionalLadderNetworks
from accelbrainbase.observabledata._mxnet.convolutionalneuralnetworks.convolutionalautoencoder.contractive_cae import ContractiveCAE

from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.convolutional_neural_networks import ConvolutionalNeuralNetworks
from accelbrainbase.observabledata._mxnet.neural_networks import NeuralNetworks

from accelbrainbase.observabledata._mxnet.adversarialmodel.discriminative_model import DiscriminativeModel
from accelbrainbase.observabledata._mxnet.adversarialmodel.generative_model import GenerativeModel
from accelbrainbase.computableloss._mxnet.generator_loss import GeneratorLoss
from accelbrainbase.computableloss._mxnet.discriminator_loss import DiscriminatorLoss
from accelbrainbase.samplabledata.true_sampler import TrueSampler
from accelbrainbase.samplabledata.condition_sampler import ConditionSampler
from accelbrainbase.samplabledata.noisesampler._mxnet.uniform_noise_sampler import UniformNoiseSampler
from accelbrainbase.controllablemodel._mxnet.gan_controller import GANController

from accelbrainbase.computableloss._mxnet.l2_norm_loss import L2NormLoss
from accelbrainbase.extractabledata.unlabeled_csv_extractor import UnlabeledCSVExtractor
from accelbrainbase.iteratabledata._mxnet.unlabeled_sequential_csv_iterator import UnlabeledSequentialCSVIterator
from accelbrainbase.noiseabledata._mxnet.gauss_noise import GaussNoise
from accelbrainbase.observabledata._mxnet.lstm_networks import LSTMNetworks
from accelbrainbase.observabledata._mxnet.lstmnetworks.encoder_decoder import EncoderDecoder

import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
import pandas as pd
from mxnet.gluon.nn import Conv2D
from mxnet.gluon.nn import Conv2DTranspose
from mxnet.gluon.nn import BatchNorm

from algowars.extractablehistoricaldata.facade_alpha_vantage import FacadeAlphaVantage



class VolatilityGAN(object):
    '''
    '''
    # Target features
    __target_features_list = [
        "adjusted_close",
    ]

    def get_target_features_list(self):
        ''' getter '''
        return self.__target_features_list
    
    def set_target_features_list(self, value):
        ''' setter '''
        self.__target_features_list = value

    target_features_list = property(get_target_features_list, set_target_features_list)

    def __init__(
        self,
        extractable_historical_data,
        ticker_list,
        start_date,
        end_date,
        batch_size=50,
        seq_len=30,
        learning_rate=1e-04,
        target_features_list=None,
        ctx=mx.gpu(),
        g_params_path="params/recursive_seq_2_seq_model_re_encoder_model.params",
        re_e_params_path="params/recursive_seq_2_seq_model_model.params",
        d_params_path="params/discriminative_model_model.params",
        transfer_flag=True,
        diff_mode=True,
        log_mode=True,
        multi_fractal_mode=True,
        long_term_seq_len=30,
    ):
        '''
        Init.

        Args:
            extractable_historical_data:        is-a `ExtractableHistoricalData`.
            ticker_list:                        `list` if tickers.
            start_date:                         `str` of start date.
            end_date:                           `str` of end date.
            batch_size:                         `int` of batch size.
            seq_len:                            `int` of the length of sequence.
            learning_rate:                      `float` of learning rate.
            target_features_list:               `list` of `str`. The value is ...
                                                    - `adjusted_close`: adjusted close.
                                                    - `close`: close.
                                                    - `high`: high value.
                                                    - `low`: low value.
                                                    - `open`: open.
                                                    - `volume`: volume.

            ctx:                                `mx.gpu()` or `mx.cpu()`.
            g_params_path:                      `str` of path to generator's learned parameters.
            re_e_params_path:                   `str` of path to re-encoder's learned parameters.
            d_params_path:                      `str` of path to discriminator's learned parameters.
            transfer_flag:                      `bool`. If `True`, this class will do transfer learning.
            multi_fractal_mode:                 `bool`.
            long_term_seq_len:                  `int`.
        '''
        if isinstance(extractable_historical_data, ExtractableHistoricalData) is False:
            raise TypeError()
        self.__extractable_historical_data = extractable_historical_data

        if target_features_list is None:
            target_features_list = self.__target_features_list

        computable_loss = L2NormLoss()
        initializer = mx.initializer.Uniform(0.3)
        hybridize_flag = True

        volatility_conditional_true_sampler = VolatilityConditionalTrueSampler(
            extractable_historical_data=extractable_historical_data,
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size, 
            seq_len=seq_len, 
            channel=len(ticker_list),
            target_features_list=target_features_list,
            diff_mode=diff_mode,
            log_mode=log_mode,
            ctx=ctx,
            lstm_mode=True,
            expand_dims_flag=False
        )

        volatility_conditional_noise_sampler = VolatilityConditionalNoiseSampler(
            extractable_historical_data=extractable_historical_data,
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size, 
            seq_len=seq_len, 
            channel=len(ticker_list),
            target_features_list=target_features_list,
            diff_mode=diff_mode,
            log_mode=log_mode,
            ctx=ctx,
            lstm_mode=True
        )

        if multi_fractal_mode is True:
            _Seq2Seq = MultiFractalSeq2SeqModel
        else:
            _Seq2Seq = RecursiveSeq2SeqModel
            print("not multi-fractal mode.")

        generative_model = _Seq2Seq(
            batch_size=batch_size,
            seq_len=seq_len,
            output_n=len(ticker_list),
            hidden_n=len(ticker_list),
            noise_sampler=volatility_conditional_noise_sampler, 
            model=None, 
            initializer=None,
            computable_loss=None,
            condition_sampler=None,
            conditonal_dim=1,
            learning_rate=learning_rate,
            optimizer_name="SGD",
            hybridize_flag=hybridize_flag,
            scale=1.0, 
            ctx=ctx, 
            channel=len(ticker_list),
            diff_mode=diff_mode,
            log_mode=log_mode,
            expand_dims_flag=False
        )

        generative_model.long_term_seq_len = long_term_seq_len

        o_act = "sigmoid"

        d_model = LSTMNetworks(
            # is-a `ComputableLoss` or `mxnet.gluon.loss`.
            computable_loss=computable_loss,
            # `int` of batch size.
            batch_size=batch_size,
            # `int` of the length of series.
            seq_len=seq_len*2,
            # `int` of the number of units in hidden layer.
            hidden_n=len(ticker_list),
            # `int` of the number of units in output layer.
            output_n=1,
            # `float` of dropout rate.
            dropout_rate=0.0,
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` 
            # that activates observed data points.
            observed_activation="tanh",
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in input gate.
            input_gate_activation="sigmoid",
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in forget gate.
            forget_gate_activation="sigmoid",
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output gate.
            output_gate_activation="sigmoid",
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in hidden layer.
            hidden_activation="tanh",
            # `act_type` in `mxnet.ndarray.Activation` or `mxnet.symbol.Activation` in output layer.
            output_activation=o_act,
            # `bool` that means this class has output layer or not.
            output_layer_flag=True,
            # `bool` for using bias or not in output layer(last hidden layer).
            output_no_bias_flag=False,
            # Call `mxnet.gluon.HybridBlock.hybridize()` or not.
            hybridize_flag=hybridize_flag,
            # `mx.cpu()` or `mx.gpu()`.
            ctx=ctx,
            input_adjusted_flag=False
        )

        discriminative_model = DiscriminativeModel(
            model=d_model, 
            learning_rate=learning_rate,
            optimizer_name="SGD",
            hybridize_flag=hybridize_flag,
            scale=1.0, 
            ctx=ctx, 
        )

        if transfer_flag is True:
            if g_params_path is not None:
                try:
                    generative_model.re_encoder_model.load_parameters(g_params_path)
                except:
                    print("generative_model.re_encoder_model.load_parameters is false.")
            if re_e_params_path is not None:
                try:
                    generative_model.model.load_parameters(re_e_params_path)
                except:
                    print("generative_model.model.load_parameters is false.")
            if d_params_path is not None:
                try:
                    discriminative_model.model.load_parameters(d_params_path)
                except:
                    print("discriminative_model.model.load_parameters is false.")

        GAN = VolatilityGANController(
            true_sampler=volatility_conditional_true_sampler,
            generative_model=generative_model,
            discriminative_model=discriminative_model,
            generator_loss=GeneratorLoss(weight=1.0),
            discriminator_loss=DiscriminatorLoss(weight=1.0),
            feature_matching_loss=L2NormLoss(weight=1.0),
            mean_regression_loss=L2NormLoss(weight=0.01),
            mean_regression_weight=1.0,
            similar_loss=L2NormLoss(weight=1.0),
            similar_weight=1.0,
            optimizer_name="SGD",
            learning_rate=learning_rate,
            learning_attenuate_rate=1.0,
            attenuate_epoch=50,
            hybridize_flag=hybridize_flag,
            scale=1.0,
            ctx=ctx,
            initializer=initializer,
        )

        self.__diff_mode = diff_mode
        self.__noise_sampler = volatility_conditional_noise_sampler
        self.__true_sampler = volatility_conditional_true_sampler
        self.__generative_model = generative_model
        self.__discriminative_model = discriminative_model
        self.__GAN = GAN
        self.__ticker_list = ticker_list
        self.__seq_len = seq_len
        self.__log_mode = log_mode
        self.__g_params_path = g_params_path
        self.__re_e_params_path = re_e_params_path
        self.__d_params_path = d_params_path

    def learn(self, iter_n=500, k_step=10):
        '''
        Learning.

        Args:
            iter_n:     The number of training iterations.
            k_step:     The number of learning of the `discriminator`.

        '''
        self.__GAN.learn(iter_n=iter_n, k_step=k_step)
        self.__generative_model = self.__GAN.generative_model
        self.__discriminative_model = self.__GAN.discriminative_model

    def extract_logs(self):
        '''
        Extract update logs data.

        Returns:
            The shape is:
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `discriminator`'s update turn.
            - `list` of probabilities inferenced by the `discriminator` (mean) in the `generator`'s update turn.

        '''
        d_logs_list = self.__GAN.discriminative_loss_arr.tolist()
        g_logs_list = self.__GAN.generative_loss_arr.tolist()
        return (
            d_logs_list,
            g_logs_list,
        )

    def extract_feature_matching_logs(self):
        return self.__GAN.feature_matching_loss_arr

    def inference(
        self,
        generated_start_date,
        generated_end_date,
        ticker_list,
        pre_delta_day_n=7,
    ):
        true_start_date = datetime.strptime(generated_start_date, "%Y-%m-%d") - timedelta(days=pre_delta_day_n)
        true_start_date = datetime.strftime(true_start_date, "%Y-%m-%d")
        pre_stock_df = self.__extractable_historical_data.extract(
            true_start_date,
            generated_start_date,
            self.__ticker_list
        )

        true_stock_df = self.__extractable_historical_data.extract(
            generated_start_date,
            generated_end_date,
            self.__ticker_list
        )

        g_s_d = datetime.strptime(generated_start_date, "%Y-%m-%d")
        g_e_d = datetime.strptime(generated_end_date, "%Y-%m-%d")
        diff_g = (g_e_d - g_s_d).days
        if diff_g > self.__seq_len:
            generated_n = diff_g // self.__seq_len
        else:
            generated_n = 1

        generated_arr, rest_generated_arr = self.__generate_volatility(
            start_date=generated_start_date,
            end_date=generated_end_date, 
            limit=generated_n
        )

        generated_stock_df_list = self.__recursive_generate(
            generated_arr=generated_arr, 
            start_date=generated_start_date
        )

        rest_generated_stock_df_list = self.__recursive_generate(
            generated_arr=rest_generated_arr, 
            start_date=generated_end_date
        )

        _true_stock_df = true_stock_df[
            [col for col in true_stock_df.columns if col not in self.__target_features_list]
        ]

        for i in range(len(generated_stock_df_list)):
            pre_df = pre_stock_df.copy()
            pre_df["batch_n"] = str(i)

            generated_stock_df_list[i]["batch_n"] = str(i)
            rest_generated_stock_df_list[i]["batch_n"] = str(i)
            df_list = []
            rest_df_list = []
            for ticker in self.__ticker_list:
                df = generated_stock_df_list[i][generated_stock_df_list[i].ticker == ticker]
                rest_df = rest_generated_stock_df_list[i][rest_generated_stock_df_list[i].ticker == ticker]
                true_sub_df = _true_stock_df[_true_stock_df.ticker == ticker]

                if df.shape[0] > true_sub_df.shape[0]:
                    df = df.iloc[:true_sub_df.shape[0]]
                
                df = df.reset_index()
                true_sub_df = true_sub_df[[col for col in true_sub_df.columns if col not in df.columns]]
                true_sub_df = true_sub_df.reset_index()

                for target_feature in self.__target_features_list:
                    df[target_feature] = df[target_feature] - df[target_feature].mean()
                    if self.__log_mode is True:
                        df[target_feature] = df[target_feature].apply(np.exp)
                    rest_df[target_feature] = rest_df[target_feature] - rest_df[target_feature].mean()
                    if self.__log_mode is True:
                        rest_df[target_feature] = rest_df[target_feature].apply(np.exp)

                df_list.append(df)
                rest_df_list.append(rest_df)

            generated_stock_df_list[i] = pd.concat(df_list)
            rest_generated_stock_df_list[i] = pd.concat(rest_df_list)

            df_list = []
            rest_df_list = []

            for ticker in self.__ticker_list:
                df = generated_stock_df_list[i][generated_stock_df_list[i].ticker == ticker]
                rest_df = rest_generated_stock_df_list[i][rest_generated_stock_df_list[i].ticker == ticker]

                df = pd.concat([df, true_sub_df], axis=1)
                pre_t_df = pre_df[pre_df.ticker == ticker]
                df = df[[col for col in pre_t_df.columns]]
                df = pd.concat([pre_t_df, df])
                df = df.iloc[pre_t_df.shape[0] - 1:]

                for target_feature in self.__target_features_list:
                    df.loc[:, target_feature] = df[target_feature].fillna(method='ffill')
                    df.loc[:, target_feature] = df[target_feature].fillna(method='bfill')
                    if self.__diff_mode is True and self.__log_mode is True:
                        df.loc[:, target_feature] = df[target_feature].cumprod()
                    elif self.__diff_mode is True and self.__log_mode is False:
                        df.loc[:, target_feature] = df[target_feature].cumsum()

                _df = df[[col for col in rest_df.columns]]
                rest_df = pd.concat([_df, rest_df])
                rest_df = rest_df.iloc[_df.shape[0] - 1:]
                for target_feature in self.__target_features_list:
                    rest_df.loc[:, target_feature] = rest_df[target_feature].fillna(method='ffill')
                    rest_df.loc[:, target_feature] = rest_df[target_feature].fillna(method='bfill')
                    if self.__diff_mode is True and self.__log_mode is True:
                        rest_df.loc[:, target_feature] = rest_df[target_feature].cumprod()
                    elif self.__diff_mode is True and self.__log_mode is False:
                        rest_df.loc[:, target_feature] = rest_df[target_feature].cumsum()

                df_list.append(df.iloc[1:, :])
                rest_df_list.append(rest_df.iloc[1:, :])

            generated_stock_df_list[i] = pd.concat(df_list)
            rest_generated_stock_df_list[i] = pd.concat(rest_df_list)

        return generated_stock_df_list, true_stock_df, rest_generated_stock_df_list

    def save_parameters(
        self,
        g_params_path=None,
        re_e_params_path=None,
        d_params_path=None
    ):
        if g_params_path is not None:
            self.__generative_model.re_encoder_model.save_parameters(g_params_path)
        if re_e_params_path is not None:
            self.__generative_model.model.save_parameters(re_e_params_path)
        if d_params_path is not None:
            self.__discriminative_model.model.save_parameters(d_params_path)

    def __generate_volatility(self, start_date, end_date, limit=1):
        '''
        Generate stock prices from volatility.

        Args:
            start_date:     Start date.
            end_date:       End date.
            limit:          The number of generations.
        
        Returns:
            `np.ndarray`, `np.ndarray`
        '''
        if self.__generative_model.noise_sampler.start_date != start_date or self.__generative_model.noise_sampler.end_date != end_date:
            self.__generative_model.noise_sampler.start_date = start_date
            self.__generative_model.noise_sampler.end_date = end_date
            self.__generative_model.noise_sampler.setup_data()
        self.__generative_model.noise_sampler.fix_date_flag = True

        generative_arr_list = self.__generative_model.recursive_draw(limit=limit)

        generative_arr = None
        for i in range(len(generative_arr_list)):
            if generative_arr is None:
                generative_arr = generative_arr_list[i].asnumpy()
            else:
                generative_arr = np.concatenate((generative_arr, generative_arr_list[i].asnumpy()), axis=1)

        rest_generated_arr_list = self.__generative_model.rest_recursive_draw(limit=12)

        rest_generated_arr = None
        for i in range(len(rest_generated_arr_list)):
            if rest_generated_arr is None:
                rest_generated_arr = rest_generated_arr_list[i].asnumpy()
            else:
                rest_generated_arr = np.concatenate((rest_generated_arr, rest_generated_arr_list[i].asnumpy()), axis=1)

        return generative_arr, rest_generated_arr

    def __recursive_generate(self, generated_arr, start_date):
        stock_df_list = []
        for i in range(generated_arr.shape[0]):
            seq_date_df = pd.date_range(
                start=start_date,
                periods=generated_arr[i].shape[0]
            )
            stock_df = self.__convert_sample_into_stock_df(
                sampled_arr=generated_arr[i],
                seq_date_df=seq_date_df
            )
            stock_df_list.append(stock_df)
        
        return stock_df_list

    def __convert_sample_into_stock_df(self, sampled_arr, seq_date_df):
        _ticker_list = []
        feature_arr_list = []
        for ticker_key in range(len(self.__ticker_list)):
            ticker = self.__ticker_list[ticker_key]
            for seq in range(seq_date_df.shape[0]):
                _ticker_list.append(ticker)
                date = seq_date_df[seq]
                feature_arr = sampled_arr[seq, ticker_key]
                feature_arr_list.append(feature_arr)

        feature_df = pd.DataFrame(feature_arr_list, columns=self.__target_features_list)
        ticker_df = pd.DataFrame(_ticker_list, columns=["ticker"])
        stock_df = pd.concat([ticker_df, feature_df], axis=1)

        return stock_df


def Main(params_dict):
    '''
    Entry points.
    
    Args:
        params_dict:    `dict` of parameters.
    '''
    from algowars.extractablehistoricaldata.facade_alpha_vantage import FacadeAlphaVantage
    # not logging but logger.
    from logging import getLogger, StreamHandler, NullHandler, DEBUG, ERROR

    import tracemalloc
    tracemalloc.start()

    with open(params_dict["api_key"], "r") as f:
        params_dict["api_key"] = f.read()
        f.close()

    if params_dict["verbose"] is True:
        print("Load historical data.")

    logger = getLogger("accelbrainbase")
    handler = StreamHandler()
    if params_dict["verbose"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)
    logger.addHandler(handler)

    logger = getLogger("pygan")
    handler = StreamHandler()
    if params_dict["verbose"] is True:
        handler.setLevel(DEBUG)
        logger.setLevel(DEBUG)
    else:
        handler.setLevel(ERROR)
        logger.setLevel(ERROR)
    logger.addHandler(handler)

    extractable_historical_data = FacadeAlphaVantage(
        api_key=params_dict["api_key"],
        logs_dir=params_dict["logs_dir"],
    )
    stock_master_df = pd.read_csv(params_dict["ticker_master_data"])

    if params_dict["stock_choiced"] != "all":
        if params_dict["stock_choiced"] == "random":
            ticker_key_arr = np.arange(stock_master_df.shape[0])
            np.random.shuffle(ticker_key_arr)
            extracted_num = np.random.randint(low=5, high=ticker_key_arr.shape[0])
            stock_master_df = stock_master_df.iloc[ticker_key_arr[:extracted_num]]
        else:
            ticker_key_list = params_dict["stock_choiced"].split(",")
            stock_master_df = stock_master_df[stock_master_df.ticker.isin(ticker_key_list)]

    ticker_list = stock_master_df.ticker.values.tolist()

    volatility_GAN = VolatilityGAN(
        extractable_historical_data=extractable_historical_data,
        ticker_list=ticker_list,
        start_date=params_dict["start_date"],
        end_date=params_dict["end_date"],
        batch_size=params_dict["batch_size"],
        seq_len=params_dict["seq_len"],
        learning_rate=params_dict["learning_rate"],
        g_params_path=params_dict["g_params_path"],
        re_e_params_path=params_dict["re_e_params_path"],
        d_params_path=params_dict["d_params_path"],
        transfer_flag=params_dict["transfer_flag"],
    )

    if params_dict["verbose"] is True:
        print("Build volatility GAN.")

    try:
        volatility_GAN.learn(iter_n=params_dict["item_n"], k_step=params_dict["k_step"])
    except KeyboardInterrupt:
        print("KeyboardInterrupt.")

    volatility_GAN.save_parameters(
        g_params_path=params_dict["g_params_path"],
        re_e_params_path=params_dict["re_e_params_path"],
        d_params_path=params_dict["d_params_path"],
    )

    d_logs_list, g_logs_list = volatility_GAN.extract_logs()
    feature_matching_arr = volatility_GAN.extract_feature_matching_logs()

    if params_dict["verbose"] is True:
        print("Training volatility AAE is end.")
        print("-" * 100)
        print("D logs:")
        print(d_logs_list[-5:])
        print("-" * 100)
        print("G logs:")
        print(g_logs_list[-5:])

    generated_stock_df_list, true_stock_df, rest_generated_stock_df_list = volatility_GAN.inference(
        params_dict["generated_start_date"],
        params_dict["generated_end_date"],
        ticker_list,
    )

    pd.concat(generated_stock_df_list).to_csv(params_dict["logs_dir"] + "generated_volatility.csv", index=False)
    pd.concat(rest_generated_stock_df_list).to_csv(params_dict["logs_dir"] + "generated_rest_volatility.csv", index=False)
    true_stock_df.to_csv(params_dict["logs_dir"] + "true_volatility.csv", index=False)
    np.save(params_dict["logs_dir"] + "d_logs", d_logs_list)
    np.save(params_dict["logs_dir"] + "g_logs", g_logs_list)
    np.save(params_dict["logs_dir"] + "feature_matching_logs", feature_matching_arr)

    print("end.")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)


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
        default="result/",
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
        default=40,
        help="batch size."
    )
    parser.add_argument(
        "-sl",
        "--seq_len",
        type=int,
        default=10,
        help="The length of sequences."
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-09,
        help="Learnign rate."
    )

    parser.add_argument(
        "-in",
        "--item_n",
        type=int,
        default=5000,
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
        "-tf",
        "--transfer_flag",
        action="store_true",
        default=False,
        help="Do transfer learning or not."
    )
    args = parser.parse_args()
    params_dict = vars(args)

    print(params_dict)
    Main(params_dict)
