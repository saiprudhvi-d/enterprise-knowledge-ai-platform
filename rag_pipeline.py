"""
Enterprise Knowledge AI Platform
RAG Pipeline using LangChain + OpenAI + Pinecone
Supports natural language querying across 500K+ enterprise documents
"""

import os
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
import pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")
INDEX_NAME = os.getenv("PINECONE_INDEX", "enterprise-knowledge")


def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(INDEX_NAME, dimension=1536, metric="cosine")
    return pinecone.Index(INDEX_NAME)


def load_and_split_documents(file_paths: List[str], chunk_size: int = 1000) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path)
        all_docs.extend(splitter.split_documents(loader.load()))
    print(f"Loaded {len(all_docs)} document chunks.")
    return all_docs


def ingest_documents(file_paths: List[str]):
    docs = load_and_split_documents(file_paths)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Pinecone.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print(f"Ingested {len(docs)} chunks into index '{INDEX_NAME}'.")
    return vectorstore


def build_rag_chain(vectorstore, model: str = "gpt-4", k: int = 5) -> RetrievalQA:
    llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=OPENAI_API_KEY)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )


def query(chain: RetrievalQA, question: str) -> dict:
    result = chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]],
    }


if __name__ == "__main__":
    init_pinecone()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = Pinecone.from_existing_index(INDEX_NAME, embeddings)
    chain = build_rag_chain(vectorstore)
    response = query(chain, "What are the Q3 revenue highlights?")
    print("Answer:", response["answer"])
    print("Sources:", response["sources"])
