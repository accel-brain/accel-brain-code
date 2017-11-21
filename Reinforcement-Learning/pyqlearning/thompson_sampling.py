#!/user/bin/env python
# -*- coding: utf-8 -*-
from numpy.random import beta


class ThompsonSampling(object):
    '''
    Thompson Sampling

    以下のコードをヒントに記述。
    https://gist.github.com/laughing/f4ca5e2ec7ea602be938#file-ts-py

    インターフェイス仕様は以下の書籍で取り上げられている
    バンディットアルゴリズムに合わせている。

    http://shop.oreilly.com/product/0636920027393.do


    Thompson Samplingに関しては以下を参照。

    Agrawal, S., & Goyal, N. (2011). Analysis of Thompson sampling for the multi-armed bandit problem. arXiv preprint arXiv:1111.1797.
    Chapelle, O., & Li, L. (2011). An empirical evaluation of thompson sampling. In Advances in neural information processing systems (pp. 2249-2257).

    '''

    # アルファの加算値
    __alpha_add_value = 1
    # ベータの加算値
    __beta_add_value = 1
    # アルファの順序統計量
    __alpha_order_statistic = []
    # ベータの順序統計量
    __beta_order_statistic = []
    # θi～Beta(αi,βi),andI(t)＝argmaxθi
    __expected_values = []

    def get_alpha_order_statistic(self):
        '''
        getter
        アルファの順序統計量
        '''
        if isinstance(self.__alpha_order_statistic, list) is False:
            raise TypeError("The type of __alpha_order_statistic must be list.")
        return self.__alpha_order_statistic

    def set_alpha_order_statistic(self, value):
        '''
        setter
        アルファの順序統計量
        '''
        if isinstance(value, list) is False:
            raise TypeError("The type of __alpha_order_statistic must be list.")
        self.__alpha_order_statistic = value

    alpha_order_statistic = property(get_alpha_order_statistic, set_alpha_order_statistic)

    def get_beta_order_statistic(self):
        '''
        getter
        ベータの順序統計量
        '''
        if isinstance(self.__beta_order_statistic, list) is False:
            raise TypeError("The type of __beta_order_statistic must be list.")
        return self.__beta_order_statistic

    def set_beta_order_statistic(self, value):
        '''
        setter
        ベータの順序統計量
        '''
        if isinstance(value, list) is False:
            raise TypeError("The type of __beta_order_statistic must be list.")
        self.__beta_order_statistic = value

    beta_order_statistic = property(get_beta_order_statistic, set_beta_order_statistic)

    def get_expected_values(self):
        '''
        getter
        報酬期待値
        '''
        if isinstance(self.__expected_values, list) is False:
            raise TypeError("__expected_values must be list.")
        return self.__expected_values

    def set_expected_values(self, value):
        '''
        setter
        報酬期待値
        '''
        if isinstance(value, list) is False:
            raise TypeError("__expected_values must be list.")
        self.__expected_values = value

    expected_values = property(get_expected_values, set_expected_values)

    def __init__(self, alpha_order_statistic=[], beta_order_statistic=[], expected_values=[]):
        '''
        インスタンス化し、各初期値をセットアップする

        Args:
            alpha_order_statistic:      アルファの順序統計量の初期値
            beta_order_statistic:       ベータの順序統計量の初期値
            expected_values:            報酬期待値の初期値
        '''
        self.alpha_order_statistic = alpha_order_statistic
        self.beta_order_statistic = beta_order_statistic
        self.expected_values = expected_values

    def initialize(self, n_arms):
        '''
        バンディットアルゴリズムの腕の数に応じ、各統計量や期待値を初期化する
        オライリー本準拠。

        Args:
            n_arms:     バンディットアルゴリズムの腕の数
        '''
        self.alpha_order_statistic = [0 for col in range(n_arms)]
        self.beta_order_statistic = [0 for col in range(n_arms)]
        self.expected_values = [0.0 for col in range(n_arms)]

    def thetalize(self):
        '''
        共役事前分布のベータ分布を前提として事後分布を求める。

        Returns:
            各腕の事後分布と報酬期待値のtupleを格納したリスト
        '''
        if len(self.alpha_order_statistic) != len(self.beta_order_statistic):
            raise Exception("alpha and beta is not same count.")

        theta = []
        for arm in range(len(self.alpha_order_statistic)):
            beta_dist = beta(
                self.alpha_order_statistic[arm] + self.__alpha_add_value,
                self.beta_order_statistic[arm] + self.__beta_add_value
            )
            theta.append((arm, beta_dist, self.expected_values[arm]))
        return theta

    def select_arm(self):
        '''
        腕を引く

        Return:
            事後分布から事後分布が最大となる腕の順序統計量のキー
        '''
        theta = self.thetalize()
        theta_max = [theta[i][1] for i in range(len(theta))]

        return theta[theta_max.index(max(theta_max))][0]

    def select_arm_ranked(self, limit=30):
        '''
        事後分布と報酬期待値の積が大きい順に腕のキーをリスト化して返す

        Args:
            limit:      腕の数の最大値

        Returns:
            事後分布と報酬期待値の積が大きい順に腕のキーを格納したリスト

        '''
        theta = self.thetalize()
        if len(theta) < limit:
            raise Exception("len(theta) < limit")

        result = []
        counter = 0
        while counter < limit:
            theta_max = [theta[i][1] * theta[i][2] for i in range(len(theta))]
            result.append(theta[theta_max.index(max(theta_max))])
            counter += 1
            del theta[theta_max.index(max(theta_max))]

        return result

    def update(self, chosen_arm, reward):
        '''
        バンディットアルゴリズムの更新処理

        Args:
            chosen_arm:     腕のキー
            reward:         ベルヌーイ分布

        '''
        if reward == 1:
            self.alpha_order_statistic[chosen_arm] += 1
        else:
            self.beta_order_statistic[chosen_arm] += 1

        n = float(self.alpha_order_statistic[chosen_arm]) + self.beta_order_statistic[chosen_arm]
        self.expected_values[chosen_arm] = (n - 1) / n * self.expected_values[chosen_arm] + 1 / n * reward
