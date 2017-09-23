# -*- coding: utf-8 -*-
from pysummarization.readable_web_pdf import ReadableWebPDF
from time import sleep
import urllib
from pyquery import PyQuery as pq


class WebScraping(object):
    '''
    Object of Web-scraping.

    This is only a demo.
    '''

    # List of scraped dom objects.
    __dom_object_list = ["body"]
    # List of not scraped dom objects.
    __remove_object_list = ["script", "style"]

    # Object of ReadableWebPdf.
    __readable_web_pdf = None

    def get_readable_web_pdf(self):
        ''' getter '''
        if isinstance(self.__readable_web_pdf, ReadableWebPDF) is False and self.__readable_web_pdf is not None:
            raise TypeError("The type of __readable_web_pdf must be ReadableWebPDF.")
        return self.__readable_web_pdf

    def set_readable_web_pdf(self, value):
        ''' setter '''
        if isinstance(value, ReadableWebPDF) is False and value is not None:
            raise TypeError("The type of __readable_web_pdf must be ReadableWebPDF.")
        self.__readable_web_pdf = value

    readable_web_pdf = property(get_readable_web_pdf, set_readable_web_pdf)

    def scrape(self, url):
        '''
        Execute Web-Scraping.
        The target dom objects are in self.__dom_object_list.

        Args:
            url:    Web site url.

        Returns:
            The result. this is a string.

        @TODO(chimera0): check URLs format.
        '''
        if isinstance(url, str) is False:
            raise TypeError("The type of url must be str.")

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
