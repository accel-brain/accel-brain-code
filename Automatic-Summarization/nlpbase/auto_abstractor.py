import nltk
from nlp_base import NlpBase
from abstractable_doc import AbstractableDoc


class AutoAbstractor(NlpBase):
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
