import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import OPENAI_API_KEY
import os
import tempfile
import openai
from typing import Any
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredWordDocumentLoader
)
import logging
import pathlib
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.schema import Document, BaseRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler

# openai settings 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

class DocumentLoader(object):
    """
        Loads in a document with a supported extendsion.
    """
    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader
    }



def load_document(temp_filepath: str) -> list[Document]: 
    """ 
        해당 펑션은 인풋으로 받은 파일 경로를 통해 파일을 로드하고 Document 리스트를 반환한다.
    """
    extension = pathlib.Path(temp_filepath).suffix
    loader = DocumentLoader.supported_extensions.get(extension)
    
    if not loader:
        raise DocumentLoaderException(
            f"Invalid extension type {extension}, cannot load this type of file"
        )

    docs = loader(temp_filepath).load()
    logging.info(docs)

    print('Loading document successful..!')
          
    return docs

def configure_retriever(docs: list[Document]) -> BaseRetriever:

    """ 
        해당 펑션은 인풋으로 받은 Document 리스트를 이용해 Retriever를 반환한다.
        Splitter와 EmbeddingModel을 Configure할 수 있다.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    
    splitted_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 2, "fetch_k": 4}
    )

    print('Configuring retriever successful..!')

    return retriever

def configure_generator()->object:

    """ 
        해당 펑션은 generator를 반환한다.
    """

    # LLM
    # set the temperature low to make sure to continuously insepct the hallucinations
    generator = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,
        streaming=True
    )
    
    return generator

def configure_chain(retriever: BaseRetriever, generator, max_tokens_limit) -> Chain:

    """ 
        해당 펑션은 인풋으로 받은 retriever과  반환한다.
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # max_tokens_limit
    chain = ConversationalRetrievalChain.from_llm(
        generator,
        retriever=retriever,
        memory=memory,
        verbose=True,
        max_tokens_limit=4000
    )

    print('Configuring chain successful..! Invoke query with chain.invoke()')

    return chain

def configure_qa_chain(uploaded_files):
    """
        read files and configure retriever and the chain
    """
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    
    for file in uploaded_files:
        print('file: ', file)
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.read())
        docs.extend(load_document(temp_filepath))

    retriever = configure_retriever(docs=docs)
    generator = configure_generator()
    

    return configure_chain(retriever=retriever, generator=generator, max_tokens_limit=3000)

if __name__ == '__main__':

    # UI 초기화
    #st.set_page_config(page_title="Bluemoose:Chat with Documents", page_icon="B")
    st.title('Bluemoose Application')
    
    uploaded_files = st.sidebar.file_uploader(
        label="파일 업로드",
        type=list(DocumentLoader.supported_extensions.keys()),
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("계속 진행하려면 도큐먼트를 업로드하세요.")
        st.stop()

    qa_chain = configure_qa_chain(uploaded_files)
    assistant = st.chat_message("assistant")
    user_query = st.chat_input(placeholder="무엇이든 물어보세요.")

    # user query 가 존재할 경우
    if user_query:
        stream_handler = StreamlitCallbackHandler(assistant)

        response = ''
        if ("조심" in user_query or "분리" in user_query):
            st.markdown("독소조항과 관련된 정보는 현재 준비중입니다.")
        else:
            response = qa_chain.run(user_query, callbacks=[stream_handler])

        st.markdown(response)


    


