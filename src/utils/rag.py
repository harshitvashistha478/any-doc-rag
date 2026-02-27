from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Collection, MilvusException, connections, db, utility
from typing import List, Any
from dotenv import load_dotenv
import os

load_dotenv()

UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")


embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

def load_data(path: str = '../../../{UPLOAD_DIRECTORY}'):
    loader = DirectoryLoader(
        f"../../../{UPLOAD_DIRECTORY}",
        glob="**/*",
        show_progress=True
    )

    data = loader.load()
    return data


def split_docs(data: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    chunks = splitter.split_documents(data)
    return chunks


def save_to_vectorDB(embedded_data: Any):
    pass
