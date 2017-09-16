# -*- coding: utf-8 -*-
from nlpbase.auto_abstractor import AutoAbstractor
from web_scraping import WebScraping
from abstractabledoc.std_abstractor import StdAbstractor
from abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from readablewebpdf.web_pdf_reading import WebPDFReading


def Main(url):
    web_scrape = WebScraping()
    web_scrape.readable_web_pdf = WebPDFReading()
    document = web_scrape.scrape(url)

    auto_abstractor = AutoAbstractor()
    abstractable_doc = TopNRankAbstractor()
    result_list = auto_abstractor.summarize(document, abstractable_doc)
    [print(sentence) for sentence in result_list["summarize_result"]]

if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    Main(url)
