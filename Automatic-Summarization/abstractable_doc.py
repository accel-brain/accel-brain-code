# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class AbstractableDoc(metaclass=ABCMeta):
    '''
    文書の自動要約アルゴリズム。

    文書要約で不要となるトークンのフィルタリングのアルゴリズムが複数個想定されるため、
    GoFに由来する委譲型のStrategy Patternで実装しておく。

    そのためこの抽象クラスは「事実上の」インターフェイスとして再利用させる。

    以下の文献に準拠する。
    Luhn, Hans Peter. "The automatic creation of literature abstracts." 
    IBM Journal of research and development 2.2 (1958): 159-165.

    Matthew A. Russell　著、佐藤 敏紀、瀬戸口 光宏、原川 浩一　監訳、長尾 高弘　訳
    『入門 ソーシャルデータ 第2版――ソーシャルウェブのデータマイニング』 2014年06月 発行
    URL：http://www.oreilly.co.jp/books/9784873116792/

    ただしオライリー本はpython2で英語の文書を対象としていて、
    尚且つ掲載されているサンプルコードもリファクタリングの余地のある内容であったため、
    再設計する。
    '''

    @abstractmethod
    def filter(self, scored_list):
        '''
        フィルタリングを実行する。

        標準偏差やTOP N Rankなど、フィルタリングの具体的な実装は下位クラスに任せる。

        Args:
            scored_list:    文章ごとの重要度を近接した文で頻出する度合いで点数化・スコアリングした内容。

        Retruns:
            引数として入力したリスト型のデータを対象に、
            フィルタリングした結果をリスト型で返す。

        '''
        raise NotImplementedError("This method must be implemented.")
