import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

VECTOR_DIR = "./ai_hub"

class Voting:
    def __init__(self, vector_store):
        """
        :param vector_store: 벡터 데이터베이스 
        """
        self.chromaDb = Chroma(persist_directory=vector_store,
                               embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

    def voting(self, sentence) -> str:
        """
        입력된 문장을 검색(Similarity Search)
        :param sentence: 검색할 문장
        :return: 유리, 불리
        """

        docs = self.chromaDb.similarity_search(sentence, k=1)
        doc = docs[0]  # page_content, metadata

        return doc.metadata['class']


class BuildVector:
    def __init__(self, vector_store):
        """
        :param vector_store: 벡터 데이터베이스 
        """
        self.chromaDb = Chroma(persist_directory=vector_store,
                               embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

        self.text_splitter = RecursiveCharacterTextSplitter(
            # 청크 크기를 매우 작게 설정합니다. 예시를 위한 설정입니다.
            chunk_size=1000,
            # 청크 간의 중복되는 문자 수를 설정합니다.
            chunk_overlap=0,
            # 문자열 길이를 계산하는 함수를 지정합니다.
            length_function=len,
            # 구분자로 정규식을 사용할지 여부를 설정합니다.
            is_separator_regex=False,
        )

    def load_xml(self, file_path: str) -> str:
        """
        지정된 XML 파일을 읽어 'cn' 엘리먼트의 텍스트를 반환한다.
        :param file_path: XML 파일 경로
        :return:
        """
        try:
            print(file_path)
            tree = ET.parse(file_path)
            root = tree.getroot()
            return root.find('.//cn').text.strip()
        except Exception:
            return None

    def embedding(self, data_folder: str, meta_class: str) -> None:
        """
        :param data_folder: XML 파일이 저장된 폴더
        :param meta_class: 메타 데이터에 저장된다, '유리' 또는 '불리'
        :return:
        """
        # 폴더 내의 파일 목록 읽기
        file_list = os.listdir(data_folder)
        # 파일 목록 출력
        for file_name in file_list:
            full_path = os.path.join(data_folder, file_name)
            sentence = self.load_xml(full_path)

            if sentence is not None:
                # 문자열을 지정된 크기의 청크로 분할한다.
                docs = self.text_splitter.create_documents([sentence])
                for page in docs:
                    page.metadata = {'class': meta_class}

                # 분할된 청크를 임베딩 데이터베이스에 저장한다.
                self.chromaDb.add_documents(docs)
                self.chromaDb.persist()


def build_vector():
    build_vector = BuildVector(VECTOR_DIR)

    # 약관이 '유리'인 데이터 임베딩
    build_vector.embedding('C:/hacker/약관/01.유리', '유리')

    # 약관이 '불리'인 데이터 임베딩
    build_vector.embedding('C:/hacker/약관/02.불리', '유리')


def test_votin():
    def load_xml(file_path: str) -> str:
        """
        지정된 XML 파일을 읽어 'cn' 엘리먼트의 텍스트를 반환한다.
        :param file_path: XML 파일 경로
        :return:
        """
        try:
            print(file_path)
            tree = ET.parse(file_path)
            root = tree.getroot()
            return root.find('.//cn').text.strip()
        except Exception:
            return None

    data = load_xml('012_개인정보취급방침_가공.xml')
    voting = Voting(VECTOR_DIR)
    rs = voting.voting(data)
    print(f'@.@ 투표 결과: {rs}')


if __name__ == '__main__':
    pass
    # 임베딩 벡터 만들기
    # build_vector()
    # 임베딩 벡터 테스트
    test_votin()
