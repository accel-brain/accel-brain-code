from abstractable_doc import AbstractableDoc


class TopNRankAbstractor(AbstractableDoc):
    '''
    トップNにランク付けされたトークンだけを返す。
    '''

    # トップNのNの値
    __top_n = 10

    def get_top_n(self):
        '''
        getter
        委譲先でメソッドが実行された際に参照される。

        Returns:
            トップNのNの値を数値型で返す。

        '''
        if isinstance(self.__top_n, int) is False:
            raise TypeError("The type of __top_n must be int.")
        return self.__top_n

    def set_top_n(self, value):
        '''
        setter
        デフォルトから変えたいなら委譲前に。

        Args:
            value:  トップNのNの値。

        Raises:
            TypeError: 引数に数値以外の方の変数を入力した場合に発生する。

        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __top_n must be int.")
        self.__top_n = value

    # トップNのNのプロパティ
    top_n = property(get_top_n, set_top_n)

    def filter(self, scored_list):
        '''
        TOP N Rankでフィルタリングを実行する。

        Args:
            scored_list:    文章ごとの重要度を近接した文で頻出する度合いで点数化・スコアリングした内容。

        Retruns:
            引数として入力したリスト型のデータを対象に、
            フィルタリングした結果をリスト型で返す。

        '''
        top_n_key = -1 * self.top_n
        top_n_list = sorted(scored_list, key=lambda x: x[1])[top_n_key:]
        result_list = sorted(top_n_list, key=lambda x: x[0])
        return result_list
