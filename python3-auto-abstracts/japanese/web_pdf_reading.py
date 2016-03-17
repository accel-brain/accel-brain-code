from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import urllib.request
from interface.readable_web_pdf import ReadableWebPDF


class WebPDFReading(ReadableWebPDF):
    '''
    Web上のPDFをURLを参照して読み取る
    '''

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
        path, headers = urllib.request.urlretrieve(url)
        return self.path_to_text(path)

    def path_to_text(self, path):
        '''
        ローカルに配置したPDFを読み込んで
        文字列のテキストに変換して返す

        とはいえ単純に公式ドキュメントのコードサンプルをラッピングしただけ

        #TODO(chimera0):もう少し真面目に

        Args:
            path:   ローカルの絶対パス

        Returns:
            PDFの文書内容の文字列

        '''
        #TODO(chimera0):もう少し真面目に
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()

        pages_data = PDFPage.get_pages(
            fp,
            pagenos,
            maxpages=maxpages,
            password=password,
            caching=caching,
            check_extractable=True
        )

        for page in pages_data:
            interpreter.process_page(page)

        text = retstr.getvalue()
        text = text.replace("\n", "")

        fp.close()
        device.close()
        retstr.close()
        return text

    def is_pdf_url(self, url):
        '''
        引数として入力したURL先のリソースが
        PDFか否かを判断する

        とはいえ単純に拡張子を観るだけ

        #TODO(chimera0):もう少し真面目に

        Args:
            url:    URL

        Returns:
            True: PDF, False: not PDF
        '''
        #TODO(chimera0):もう少し真面目に
        if url[-4:] == ".pdf":
            return True
        else:
            return False

if __name__ == "__main__":
    url = "hogehoge.pdf"
    text = WebPDFReading().url_to_text(url)
    print(text[:300])
