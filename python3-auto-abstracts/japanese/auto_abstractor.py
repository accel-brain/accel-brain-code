#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import MeCab
from time import sleep
from pyquery import PyQuery as pq
import urllib.request
import nltk
import numpy
from interface.readable_web_pdf import ReadableWebPDF


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


class AbstractableStd(AbstractableDoc):
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
            avg = numpy.mean([s[1] for s in scored_list])
            std = numpy.std([s[1] for s in scored_list])
        else:
            avg = 0
            std = 0
        limiter = avg + 0.5 * std
        mean_scored = [(sent_idx, score) for (sent_idx, score) in scored_list if score > limiter]
        return mean_scored


class AbstractableTopNRank(AbstractableDoc):
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


class TextMining(metaclass=ABCMeta):
    '''
    自然言語処理系の抽象基底クラス。
    別の用途で制作したクラスを再利用している。

    今回のデモ用に暫定的にコーディングした。
    nltkとMeCabをつなぎこめるようなクラス構成にしても良いかもしれない。
    '''

    # トークンのリスト
    __token = []

    def set_token(self, value):
        '''
        setter
        トークンのリスト。

        Args:
            value: トークンのリスト

        '''
        self.__token = value

    def get_token(self):
        '''
        getter

        Returns:
            トークンのリストを返す。

        '''
        return self.__token

    # トークンのリストのプロパティ
    token = property(get_token, set_token)

    def tokenize(self, data):
        '''
        形態素解析して、トークンをプロパティ：tokenにセットする。

        Args:
            形態素解析対象となる文字列。

        '''
        mt = MeCab.Tagger("-Owakati")
        wordlist = mt.parse(data)
        self.token = wordlist.rstrip(" \n").split(" ")

    def listup_sentence(self, data, counter=0):
        '''
        日本語を文ごとに区切る。
        暫定的に、ここでは「。」と「\n」で区切る。

        Args:
            data:       区切る対象となる文字列。
            counter:    再帰回数。

        Returns:
            文ごとに区切った結果として、一文ずつ格納したリスト。

        '''
        seq_list = ["。", "\n"]
        seq = seq_list[counter]
        sentence_list = []
        [sentence_list.append(sentence + seq) for sentence in data.split(seq) if sentence != ""]
        if counter + 1 < len(seq_list):
            sentence_list_r = []
            for sentence in sentence_list:
                sentence_list_r.extend(self.listup_sentence(sentence, counter+1))

            sentence_list = sentence_list_r

        return sentence_list


class AutoAbstractor(TextMining):
    '''
    文書の自動要約アルゴリズム。
    文書要約で不要となるトークンのフィルタリングのアルゴリズムが複数個想定されるため、
    GoFに由来する委譲型のStrategy Patternで実装しておく。

    '''

    # 考慮するトークンの数
    __target_n = 100

    def get_target_n(self):
        '''
        getter

        Returns:
            考慮するトークンの数の数値。

        Raises:
            TypeError: プロパティにセットされているのが数値以外の型である場合に発生する。

        '''
        if isinstance(self.__target_n, int) is False:
            raise TypeError("The type of __target_n must be int.")
        return self.__target_n

    def set_target_n(self, value):
        '''
        setter

        Args:
            value:      考慮するトークンの数の数値。

        Raise:
            TypeError:  引数が数値以外の型である場合に発生する。

        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __target_n must be int.")
        self.__target_n = value

    # 考慮するトークンの数のプロパティ
    target_n = property(get_target_n, set_target_n)

    # 考慮するトークン間の距離
    __cluster_threshold = 5

    def get_cluster_threshold(self):
        '''
        getter

        Return:
            考慮するトークンの数の数の距離の数値型。

        Raises:
            プロパティにセットされているのが数値以外の型である場合に発生する。

        '''
        if isinstance(self.__cluster_threshold, int) is False:
            raise TypeError("The type of __cluster_threshold must be int.")
        return self.__cluster_threshold

    def set_cluster_threshold(self, value):
        '''
        setter

        Args:
            value:      考慮するトークンの数

        Raises:
            引数が数値以外の型である場合に発生する。

        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __cluster_threshold must be int.")
        self.__cluster_threshold = value

    # 考慮するトークンの数のプロパティ
    cluster_threshold = property(get_cluster_threshold, set_cluster_threshold)

    # トップNの要約結果として返す数
    __top_sentences = 5

    def get_top_sentences(self):
        '''
        getter

        Returns:
            トップNの要約結果として返す数。

        Raises:
            TypeError:  プロパティにセットされているのが数値以外の型である場合に発生する。

        '''
        if isinstance(self.__top_sentences, int) is False:
            raise TypeError("The type of __top_sentences must be int.")
        return self.__top_sentences

    def set_top_sentences(self, value):
        '''
        setter

        Args:
            value:      トップNの要約結果として返す数。

        Raises:
            TypeError:  引数の型が数値以外の型である場合に発生する。

        '''
        if isinstance(value, int) is False:
            raise TypeError("The type of __top_sentences must be int.")
        self.__top_sentences = value

    # トップNの要約結果として返す数のプロパティ
    top_sentences = property(get_top_sentences, set_top_sentences)

    def summarize(self, document, Abstractor):
        '''
        文書要約を実行する。

        Args:
            document:       要約対象となる文書の文字列。
            Abstractor:     インターフェイス：AbstractableDocを実現したオブジェクト。

        Returns:
            以下の形式の辞書。

            {
                "summarize_result": "{要約結果を一文一要素として格納したリスト｝", 
                "scoring_data":     "{summarize_resultの各要素に紐付くスコアリング結果｝"
            }

        Raises:
            TypeError:      引数：documentが文字列ではない場合に発生する。
            TypeError:      引数：Abstractorの型がAbstractableDocではない場合に発生する。

        '''
        if isinstance(document, str) is False:
            raise TypeError("The type of document must be str.")

        if isinstance(Abstractor, AbstractableDoc) is False:
            raise TypeError("The type of Abstractor must be AbstractableDoc.")

        self.tokenize(document)

        words = self.token
        normalized_sentences = self.listup_sentence(document)

        fdist = nltk.FreqDist(words)

        top_n_words = [w[0] for w in fdist.items()][:self.target_n]

        scored_list = self.__closely_associated_score(normalized_sentences, top_n_words)

        filtered_list = Abstractor.filter(scored_list)

        result_list = [normalized_sentences[idx] for (idx, score) in filtered_list]

        result_dict = {
            "summarize_result": result_list,
            "scoring_data": filtered_list
        }

        return result_dict

    def __closely_associated_score(self, normalized_sentences, top_n_words):
        '''
        文章ごとの重要度を近接した文で頻出する度合いで点数化し、スコアリングする

        Args:
            normalized_sentences:   一文一要素として格納したリスト。
            top_n_words:            要約対象文の個数。返り値のリストの要素数もこれに比例。

        Returns:
            メソッド：summarizeの返り値のキー：scoring_dataと等価のデータとなる。
            もともとのオライリー本の関数からアルゴリズム的な変更は加えていない。
            （逆に言えば、まだこのメソッドは細分化や再メソッド化の余地があるということでもある。）

        '''
        scores_list = []
        sentence_idx = -1

        for sentence in normalized_sentences:
            self.tokenize(sentence)
            sentence = self.token

            sentence_idx += 1
            word_idx = []

            # 重要度の高いトークンごとにそのキーを特定していく
            for w in top_n_words:
                # 重要なトークンがどの文で頻出するのかを指標化する
                try:
                    word_idx.append(sentence.index(w))
                # トークンが文に含まれていない場合は、特に何もしない。
                except ValueError:
                    pass

            word_idx.sort()

            # 幾つかの文は、どの重要なトークンも含んでいない場合も想定できる。
            if len(word_idx) == 0:
                continue

            # 近距離の頻出トークンごとにクラスタリングする
            clusters = []
            cluster = [word_idx[0]]
            i = 1
            while i < len(word_idx):
                if word_idx[i] - word_idx[i - 1] < self.cluster_threshold:
                    cluster.append(word_idx[i])
                else:
                    clusters.append(cluster[:])
                    cluster = [word_idx[i]]
                i += 1
            clusters.append(cluster)

            # クラスタごとに点数化する
            max_cluster_score = 0
            for c in clusters:
                significant_words_in_cluster = len(c)
                total_words_in_cluster = c[-1] - c[0] + 1
                score = 1.0 * significant_words_in_cluster \
                    * significant_words_in_cluster / total_words_in_cluster

                if score > max_cluster_score:
                    max_cluster_score = score

            scores_list.append((sentence_idx, score))

        return scores_list


class Scraping(object):
    '''
    指定したURL先のWebページを対象に、
    スクレイピングを実行する。

    今回のデモ用に暫定的にコーディングした。
    '''

    __dom_object_list = ["body"]
    __remove_object_list = ["script", "style"]

    # Web上のPDFファイルを読み込むインターフェイスを実現したオブジェクト
    __readable_web_pdf = None

    def get_readable_web_pdf(self):
        if isinstance(self.__readable_web_pdf, ReadableWebPDF) is False and self.__readable_web_pdf is not None:
            raise TypeError("The type of __readable_web_pdf must be ReadableWebPDF.")
        return self.__readable_web_pdf

    def set_readable_web_pdf(self, value):
        if isinstance(value, ReadableWebPDF) is False and value is not None:
            raise TypeError("The type of __readable_web_pdf must be ReadableWebPDF.")
        self.__readable_web_pdf = value

    readable_web_pdf = property(get_readable_web_pdf, set_readable_web_pdf)

    def scrape(self, url):
        '''
        Webスクレイピングを実行する。

        Args:
            url:    スクレイピング対象となるWebページのURL。

        Returns:
            Webスクレイピング結果として取得できた文字列。
            プライベートフィールド：__dom_object_listで指定したHTMLタグ内部が
            基本的に取得される。

        Raises:
            TypeError:  URLが文字列ではない場合に発生する。
                        正規表現等でURLの形式をチェックするように改修するのも一興。

        '''
        if isinstance(url, str) is False:
            raise TypeError("url must be str.")

        if self.readable_web_pdf is not None and self.readable_web_pdf.is_pdf_url(url) is True:
            web_data = self.readable_web_pdf.url_to_text(url)
        else:
            web_data = ""
            req = urllib.request.Request(url=url)
            with urllib.request.urlopen(req) as f:
                web = f.read().decode('utf-8')
                dom = pq(web)
                [dom(remove_object).remove() for remove_object in self.__remove_object_list]

                for dom_object in self.__dom_object_list:
                    web_data += dom(dom_object).text()

        sleep(1)
        return web_data

if __name__ == '__main__':
    import sys
    from web_pdf_reading import WebPDFReading
    web_scrape = Scraping()
    web_scrape.readable_web_pdf = WebPDFReading()
    document = web_scrape.scrape(sys.argv[1])

    auto_abstractor = AutoAbstractor()
    abstractable_doc = AbstractableTopNRank()
    result_list = auto_abstractor.summarize(document, abstractable_doc)
    [print(sentence) for sentence in result_list["summarize_result"]]
