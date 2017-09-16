#!/user/bin/env python
# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class ReadableWebPDF(metaclass=ABCMeta):
    '''
    Web上のPDFを読み込んで文字列で返すインターフェイス
    '''

    @abstractmethod
    def url_to_text(self, url):
        '''
        Web上のPDFをローカルにダウンロードして、
        そのPDFを読み込んで
        文字列のテキストに変換して返す

        Args:
            url:   Web上のURL

        Returns:
            PDFの文書内容の文字列

        '''
        raise NotImplementedError("This method must be implemented.")

    @abstractmethod
    def is_pdf_url(self, url):
        '''
        引数として入力したURL先のリソースが
        PDFか否かを判断する

        Args:
            url:    URL

        Returns:
            True: PDF, False: not PDF
        '''
        raise NotImplementedError("This method must be implemented.")
