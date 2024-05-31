# pip install PyPDF2
import PyPDF2

class PdfHolder:
    def __init__(self, pdf_path):
        self.pages = []
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        for page in range(len(pdf_reader.pages)):
            self.pages.append({
                'page': page,
                'text': pdf_reader.pages[page].extract_text()
            })

    def pageCount(self):
        """
        :return: PDF 페이지 수
        """
        return len(self.pages)

    def getPage(self, page_index):
        """
        :param page_index: 페이지 번호
        :return:
        """
        if page_index <= 0:
            page_index = 1
        return self.pages[page_index-1]


if __name__ == '__main__':
    # PDF 파일 열기
    pdf_file = open("플러스메디컬단체보험_특별약관.pdf", 'rb')
    pdf_holder = PdfHolder(pdf_file)
    pdf_file.close()

    print(f'## PAGES = {pdf_holder.pageCount()}')
    print(pdf_holder.getPage(5))
    print('*' * 80)
