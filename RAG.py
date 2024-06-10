
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
from voting import Voting, VECTOR_DIR



# openai settings 
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = os.environ["OPENAI_API_KEY"]

# initialize Voting instance
voting = Voting(VECTOR_DIR)

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
    vector_store = FAISS.from_documents(splitted_docs, embeddings)
    
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 2, "fetch_k": 4}
    )

    print('Configuring retriever successful..!')

    return retriever

def configure_generator() -> object:

    """ 
        해당 펑션은 generator(LLM)를 반환한다.
    """
    # set the temperature low to make sure to continuously insepct the hallucinations
    generator = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,
        streaming=True
    )
    
    return generator

def configure_chain(retriever: BaseRetriever, generator, max_tokens_limit = 4000) -> Chain:

    """ 
        해당 펑션은 인풋으로 받은 retriever, generator, max_tokiens_limit을 바탕으로 Chain 오브젝트를 반환한다.
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # max_tokens_limit
    chain = ConversationalRetrievalChain.from_llm(
        generator,
        retriever=retriever,
        memory=memory,
        verbose=True,
        max_tokens_limit=max_tokens_limit
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

    # Configuring QA chain
    return configure_chain(retriever=retriever, generator=generator, max_tokens_limit=3000)