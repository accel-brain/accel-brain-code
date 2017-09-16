# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import MeCab


class NlpBase(metaclass=ABCMeta):
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
