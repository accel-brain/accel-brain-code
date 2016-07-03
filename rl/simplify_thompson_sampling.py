#!/user/bin/env python
# -*- coding: utf-8 -*-


class BetaDist(object):
    '''
    ベータ分布
    二項分布の共役事前分布
    '''
    # デフォルトのα統計量
    __default_alpha = 1
    # デフォルトのβ統計量
    __default_beta = 1
    # 成功回数
    __success = 0
    # 失敗回数
    __failure = 0

    def __init__(self, default_alpha=1, default_beta=1):
        '''
        初期化

        Args:
            default_alpha:      デフォルトのα統計量
            default_beta:       デフォルトのβ統計量

        Exceptions:
            TypeError:      default_alphaが実数(int or float)ではない場合
            TypeError:      default_betaが実数(int or float)ではない場合
            ValueError:     default_alphaが正の数ではない場合
            ValueError:     default_betaが正の数ではない場合

        '''
        if isinstance(default_alpha, int) is False:
            if isinstance(default_alpha, float) is False:
                raise TypeError()
        if isinstance(default_beta, int) is False:
            if isinstance(default_beta, float) is False:
                raise TypeError()

        if default_alpha <= 0:
            raise ValueError()
        if default_beta <= 0:
            raise ValueError()

        self.__success += 0
        self.__failure += 0
        self.__default_alpha = default_alpha
        self.__default_beta = default_beta

    def observe(self, success, failure):
        '''
        正否や成否など、結果が二分される試行を観測する

        Args:
            success:      成功回数
            failure:      失敗回数

        Exceptions:
            TypeError:      successが実数(int or float)ではない場合
            TypeError:      failureが実数(int or float)ではない場合
            ValueError:     successが正の数ではない場合
            ValueError:     failureが正の数ではない場合
        '''
        if isinstance(success, int) is False:
            if isinstance(success, float) is False:
                raise TypeError()
        if isinstance(failure, int) is False:
            if isinstance(failure, float) is False:
                raise TypeError()

        if success <= 0:
            raise ValueError()
        if failure <= 0:
            raise ValueError()

        self.__success += success
        self.__failure += failure

    def likelihood(self):
        '''
        尤度を計算する

        Returns:
            尤度
        '''
        try:
            likelihood = self.__success / (self.__success + self.__failure)
        except ZeroDivisionError:
            likelihood = 0.0
        return likelihood

    def expected_value(self):
        '''
        期待値を計算する

        Returns:
            期待値
        '''
        alpha = self.__success + self.__default_alpha
        beta = self.__failure + self.__default_beta

        try:
            expected_value = alpha / (alpha + beta)
        except ZeroDivisionError:
            expected_value = 0.0
        return expected_value

    def variance(self):
        '''
        分散を計算する

        Returns:
            分散
        '''
        alpha = self.__success + self.__default_alpha
        beta = self.__failure + self.__default_beta

        try:
            variance = alpha * beta / ((alpha + beta) ** 2) * (alpha + beta + 1)
        except ZeroDivisionError:
            variance = 0.0
        return variance


class ThompsonSampling(object):
    '''
    Thompson Sampling簡略版
    期待値の高い順にリストアップするまでは良くても、
    実際には広告主やメディアの意向に応じて別の重み付け要因が絡むため、
    実用的な観点から単純に期待値の高い順にリストアップするだけに留めている
    '''

    # ベータ分布の計算オブジェクト
    __beta_dist_dict = {}

    def __init__(self, arm_id_list):
        '''
        初期化

        Args:
            arm_id_list:    腕のマスタID一覧
        '''
        [self.__beta_dist_dict.setdefault(key, BetaDist()) for key in arm_id_list]

    def pull(self, arm_id, success, failure):
        '''
        腕を引く

        Args:
            arm_id:     腕のマスタID
            success:    成功回数
            failure:    失敗回数
        '''
        self.__beta_dist_dict[arm_id].observe(success, failure)

    def recommend(self, limit=10):
        '''
        期待値の高い順に腕をリストアップする

        Args:
            limit:      リストアップ数の上限値

        Returns:
            (腕のマスタID, 期待値)のtupleのリスト
        '''
        expected_list = [(arm_id, beta_dist.expected_value()) for arm_id, beta_dist in self.__beta_dist_dict.items()]
        expected_list = sorted(expected_list, key=lambda x: x[1], reverse=True)
        return expected_list[:limit]
