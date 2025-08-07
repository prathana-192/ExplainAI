from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from models.embeddings import get_embedding_model

def get_vectorstore(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    embedding_model = get_embedding_model()
    return Chroma.from_documents(docs, embedding_model)

def get_retriever_chain(llm, vectorstore):
    return vectorstore.as_retriever()
