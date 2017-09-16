import numpy as np
from abstractable_doc import AbstractableDoc


class StdAbstractor(AbstractableDoc):
    '''
    標準偏差との差異の度合いから、
    文書要約で不要となるトークンを除去する。
    平均スコアにフィルタとしての標準偏差の半分を加算した値を利用して、
    重要ではないと見做したトークンを除去していく。
    '''

    def filter(self, scored_list):
        '''
        標準偏差を用いてフィルタリングを実行する。

        Args:
            scored_list:    文章ごとの重要度を近接した文で頻出する度合いで点数化・スコアリングした内容。

        Retruns:
            引数として入力したリスト型のデータを対象に、
            フィルタリングした結果をリスト型で返す。

        '''
        if len(scored_list) > 0:
            avg = np.mean([s[1] for s in scored_list])
            std = np.std([s[1] for s in scored_list])
        else:
            avg = 0
            std = 0
        limiter = avg + 0.5 * std
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_list if score > limiter]
        return mean_scored
