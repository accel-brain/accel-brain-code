# -*- coding: utf-8 -*-
from readable_web_pdf import ReadableWebPDF
from time import sleep
import urllib
from pyquery import PyQuery as pq


class WebScraping(object):
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
