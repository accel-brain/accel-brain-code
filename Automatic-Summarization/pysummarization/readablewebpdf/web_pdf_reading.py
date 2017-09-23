from io import StringIO
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import urllib.request
from pysummarization.readable_web_pdf import ReadableWebPDF


class WebPDFReading(ReadableWebPDF):
    '''
    Read the PDF.
    '''

    def url_to_text(self, url):
        '''
        Download PDF file and transform its document to string.

        Args:
            url:   PDF url.

        Returns:
            string.

        '''
        path, headers = urllib.request.urlretrieve(url)
        return self.path_to_text(path)

    def path_to_text(self, path):
        '''
        Transform local PDF file to string.

        Args:
            path:   path to PDF file.

        Returns:
            string.

        '''
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
        Check PDF file format.
        
        @TODO(chimera0): validation.

        Args:
            url:    URL

        Returns:
            True: PDF, False: not PDF
        '''
        if url[-4:] == ".pdf":
            return True
        else:
            return False

if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    text = WebPDFReading().url_to_text(url)
    print(text[:300])
